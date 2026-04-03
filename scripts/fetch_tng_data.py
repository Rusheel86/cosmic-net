# ============================================================
# Cosmic-Net: TNG100-1 Local Data Fetcher
# Run with: python fetch_tng_data.py
# ============================================================
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import requests
import pandas as pd
import numpy as np
import h5py
import os
import time

# ── CONFIG ──────────────────────────────────────────────────
API_KEY       = "" # Insert your TNG API key here (get from https://www.tng-project.org/api/)
BASE_URL      = "https://www.tng-project.org/api"
SIM           = "TNG100-1"
SNAPSHOT      = 99
N_HALOS       = 600  # Reduced for testing (change to 500 for full dataset)
MIN_HALO_MASS = 12.5   # log10(M_sun) for Group_M_Crit200
MIN_SUBHALOS  = 3
MAX_SUBHALOS_PER_HALO = 10  # cap satellites per halo for balanced graphs
OUTPUT_HDF5   = "data/raw/tng100_clustered.h5"
OUTPUT_CSV    = "data/raw/tng100_clustered.csv"
CHECKPOINT    = "data/raw/checkpoint.csv"
HEADERS       = {"api-key": API_KEY}

# TNG unit conversions
H_PARAM       = 0.6774   # Hubble parameter for TNG100
MASS_UNIT     = 1e10     # TNG mass unit: 1e10 M_sun/h

os.makedirs("data/raw", exist_ok=True)

# ── HELPER ──────────────────────────────────────────────────
def tng_get(url, params=None, retries=15, debug=False):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params,
                           headers=HEADERS, timeout=60)
            if debug:
                print(f"\nDEBUG: GET {url}")
                print(f"  params: {params}")
                print(f"  status: {r.status_code}")
            if r.status_code == 200:
                try:
                    data = r.json()
                    if debug:
                        print(f"  JSON keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                        if 'count' in data:
                            print(f"  count: {data.get('count')}")
                            print(f"  results: {len(data.get('results', []))}")
                    return data
                except Exception as e:
                    if debug:
                        print(f"  JSON parse error: {e}")
                        print(f"  Response text (first 500 chars): {r.text[:500]}")
                    return None
            elif r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n  Rate limited — waiting {wait}s...")
                time.sleep(wait)
            elif r.status_code == 404:
                if debug:
                    print(f"  404 Not Found")
                return None
            else:
                if debug:
                    print(f"  Error code {r.status_code}: {r.text[:200]}")
                time.sleep(5)
        except Exception as e:
            if debug:
                print(f"  Exception: {e}")
            time.sleep(5)
    return None

# ── STEP 1: Verify connection ────────────────────────────────
print("Testing connection...")
sim_info = tng_get(f"{BASE_URL}/{SIM}/")
if sim_info is None:
    raise RuntimeError("Cannot connect. Check API key.")
print(f"Connected: {sim_info['name']}")

# ── STEP 2: Get group IDs from central subhalos ─────────────────────
print(f"\nFinding top {N_HALOS} massive halos via central subhalos...")
all_centrals = []
url = f"{BASE_URL}/{SIM}/snapshots/{SNAPSHOT}/subhalos/"
params = {
    "limit"        : 100,
    "order_by"     : "-mass_log_msun",  # Get most massive centrals
    "primary_flag" : 1,  # Only central subhalos
}

while url and len(all_centrals) < N_HALOS * 2:  # Get extra in case some don't have groups
    data = tng_get(url, params=params)
    if data is None or "results" not in data:
        break
    all_centrals.extend(data["results"])
    url = data.get("next")
    params = None
    # Progress bar
    pct = min(len(all_centrals) / (N_HALOS * 2) * 100, 100)
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"\r  [{bar}] {len(all_centrals)} centrals fetched...", end="", flush=True)
    time.sleep(0.2)

print(f"\nGot {len(all_centrals)} central subhalos")

# Extract unique group IDs and fetch their masses
print("\nFetching group masses... (This requires requesting details for each subhalo, which may take a few minutes)")
group_masses = {}
failed_count = 0
for i, central in enumerate(all_centrals[:N_HALOS * 2]):
    if i % 10 == 0:
        print(f"  Checking group {i+1}/{min(len(all_centrals), N_HALOS * 2)}...", end="\r", flush=True)

    # Get group ID from central by fetching its details if needed
    group_id = central.get("grnr", None)
    if group_id is None:
        sub_d = tng_get(central["url"])
        if sub_d:
            group_id = sub_d.get("grnr")

    if group_id is None or group_id in group_masses:
        continue

    # Fetch group data to get Group_M_Crit200
    group_url = f"{BASE_URL}/{SIM}/snapshots/{SNAPSHOT}/halos/{group_id}/info.json"
    group_data = tng_get(group_url, debug=(i < 3), retries=15)  # Extended retries against gateway timeouts

    if group_data is None:
        failed_count += 1
        if i < 3:
            print(f"    Group {group_id} returned None")
        continue

    if "Group_M_Crit200" not in group_data:
        if i < 3:
            print(f"    Group {group_id} missing Group_M_Crit200, keys: {list(group_data.keys())[:10]}")
        failed_count += 1
        continue

    m_crit = group_data["Group_M_Crit200"]
    mass_linear = float(m_crit) * MASS_UNIT / H_PARAM
    mass_log = np.log10(mass_linear)
    group_masses[group_id] = {
        "Group_M_Crit200": m_crit,
        "mass_log": mass_log,
        "data": group_data
    }

    if len(group_masses) >= N_HALOS:
        break

print(f"\n  Successfully fetched {len(group_masses)} groups, {failed_count} failed")

# Sort by mass and take top N
top_groups = sorted(group_masses.items(), key=lambda x: x[1]["mass_log"], reverse=True)[:N_HALOS]
all_groups = [{"id": gid, **ginfo["data"]} for gid, ginfo in top_groups]

print(f"Selected top {len(all_groups)} massive groups")

# ── STEP 3: Fetch satellites per halo ───────────────────────
print("\nFetching satellites per halo...")
print("Progress saves every 50 halos — safe to Ctrl+C and resume")
print("NOTE: Each halo takes ~30-60 seconds (group + subhalo API calls)\n")

all_rows      = []
skipped       = 0
completed_ids = set()

# Resume from checkpoint
if os.path.exists(CHECKPOINT):
    try:
        df_ckpt       = pd.read_csv(CHECKPOINT)
        all_rows      = df_ckpt.to_dict("records")
        completed_ids = set(int(x) for x in df_ckpt["group_id"].unique())
        print(f"Resumed: {len(completed_ids)} halos, "
              f"{len(all_rows)} subhalos\n")
    except Exception:
        print("Checkpoint corrupted — starting fresh\n")

start_time = time.time()

for halo_idx, group in enumerate(all_groups):
    print(f"[{halo_idx+1}/{N_HALOS}] Processing halo...", end=" ", flush=True)

    # ETA calculation
    if halo_idx > 0 and len(completed_ids) > 0:
        elapsed   = time.time() - start_time
        per_halo  = elapsed / max(len(completed_ids), 1)
        remaining = (N_HALOS - len(completed_ids)) * per_halo
        eta_mins  = remaining / 60
        eta_str   = f"ETA: {eta_mins:.0f}m"
    else:
        eta_str = ""

    # Extract group info directly (no need for extra API call!)
    group_id = group.get("id", None)
    if group_id is None:
        skipped += 1
        print(f"SKIPPED (no group_id)")
        continue

    if group_id in completed_ids:
        print(f"SKIPPED (already done)")
        continue

    # Group_M_Crit200 is already in the group data
    group_m_crit200 = group.get("Group_M_Crit200", None)
    if group_m_crit200 is None:
        skipped += 1
        print(f"SKIPPED (no mass)")
        continue

    # Convert to M_sun: multiply by 1e10, divide by h
    halo_mass_linear = float(group_m_crit200) * MASS_UNIT / H_PARAM
    halo_mass_log = np.log10(halo_mass_linear)

    if halo_mass_log < MIN_HALO_MASS:
        skipped += 1
        print(f"SKIPPED (mass {halo_mass_log:.1f} < {MIN_HALO_MASS})")
        continue

    print(f"group={group_id}, M={halo_mass_log:.1f}...", end=" ", flush=True)

    # Pull all subhalos in this group
    sub_url    = f"{BASE_URL}/{SIM}/snapshots/{SNAPSHOT}/subhalos/"
    sub_params = {"limit": 100, "grnr": int(group_id)}
    halo_subs  = []

    while sub_url and len(halo_subs) < MAX_SUBHALOS_PER_HALO:
        sub_data = tng_get(sub_url, params=sub_params)
        if sub_data is None or "results" not in sub_data:
            break
        halo_subs.extend(sub_data["results"])
        # Trim to cap immediately
        if len(halo_subs) > MAX_SUBHALOS_PER_HALO:
            halo_subs = halo_subs[:MAX_SUBHALOS_PER_HALO]
        sub_url    = sub_data.get("next")
        sub_params = None
        time.sleep(0.03)

    if len(halo_subs) < MIN_SUBHALOS:
        skipped += 1
        continue

    # Pull detail per subhalo
    halo_rows = []
    for sub in halo_subs:
        d = tng_get(sub["url"])
        if d is None:
            continue
        try:
            def sf(key):
                v = d.get(key, np.nan)
                try:
                    return float(v) if v is not None else np.nan
                except:
                    return np.nan

            def sf_mass(key):
                v = d.get(key, 0)
                try:
                    if v is None or float(v) <= 0:
                        return np.nan
                    linear_mass = float(v) * MASS_UNIT / H_PARAM
                    return np.log10(linear_mass)
                except:
                    return np.nan

            halo_rows.append({
                "subhalo_id"       : int(sub["id"]),
                "group_id"         : int(group_id),
                "is_central"       : int(d.get("primary_flag", 0) or 0),
                "stellar_mass"     : sf_mass("mass_stars"),
                "vel_dispersion"   : sf("veldisp"),
                "half_mass_radius" : sf("halfmassrad"),
                "metallicity"      : sf("gasmetallicity"),
                "pos_x"            : sf("pos_x"),
                "pos_y"            : sf("pos_y"),
                "pos_z"            : sf("pos_z"),
                "vel_x"            : sf("vel_x"),
                "vel_y"            : sf("vel_y"),
                "vel_z"            : sf("vel_z"),
                "halo_mass"        : float(halo_mass_linear),  # linear M_sun
                "halo_mass_log"    : float(halo_mass_log),     # log10(M_sun)
            })
        except:
            continue
        time.sleep(0.03)

    if len(halo_rows) >= MIN_SUBHALOS:
        all_rows.extend(halo_rows)
        completed_ids.add(group_id)
        print(f"OK ({len(halo_rows)} subhalos) {eta_str}")
    else:
        skipped += 1
        print(f"SKIPPED (only {len(halo_rows)} subhalos)")

    # Checkpoint every 50 halos
    if len(completed_ids) % 50 == 0 and len(completed_ids) > 0:
        pd.DataFrame(all_rows).to_csv(CHECKPOINT, index=False)
        print(f"\n  ✓ Checkpoint saved at {len(completed_ids)} halos\n")

    time.sleep(0.02)

print(f"\n\nDone: {len(all_rows)} subhalos, "
      f"{len(completed_ids)} halos, {skipped} skipped")

# ── STEP 4: Clean ────────────────────────────────────────────
print("\nCleaning data...")
df = pd.DataFrame(all_rows)

if len(df) == 0:
    raise RuntimeError("No data — check API key")

critical = ["stellar_mass", "vel_dispersion",
            "half_mass_radius", "halo_mass_log"]
df = df.dropna(subset=critical)

for col in ["vel_dispersion", "half_mass_radius"]:
    df[col] = df[col].clip(lower=1e-6)
df["metallicity"] = df["metallicity"].fillna(1e-6).clip(lower=1e-6)

df["log_vel_dispersion"]   = np.log10(df["vel_dispersion"])
df["log_half_mass_radius"] = np.log10(df["half_mass_radius"])
df["log_metallicity"]      = np.log10(df["metallicity"] + 1e-6)

group_sizes  = df.groupby("group_id").size()
valid_groups = group_sizes[group_sizes >= MIN_SUBHALOS].index
df           = df[df["group_id"].isin(valid_groups)].copy()

# ── STEP 5: Save HDF5 (primary format) ─────────────────────────
print("\nSaving to HDF5 (primary format)...")
with h5py.File(OUTPUT_HDF5, 'w') as hf:
    # Store column names as attribute
    hf.attrs['columns'] = list(df.columns)
    hf.attrs['n_halos'] = int(df['group_id'].nunique())
    hf.attrs['n_subhalos'] = len(df)
    hf.attrs['source'] = 'TNG100-1'
    hf.attrs['snapshot'] = SNAPSHOT
    hf.attrs['h_param'] = H_PARAM
    hf.attrs['mass_unit'] = MASS_UNIT

    # Store each column as a dataset
    for col in df.columns:
        data = df[col].values
        if data.dtype == object:
            data = data.astype(str)
        hf.create_dataset(col, data=data, compression='gzip')

size_h5_kb = os.path.getsize(OUTPUT_HDF5) / 1024
print(f"  HDF5 saved: {OUTPUT_HDF5} ({size_h5_kb:.1f} KB)")

# ── STEP 6: Convert to CSV (secondary format) ─────────────────
print("\nConverting to CSV...")
df.to_csv(OUTPUT_CSV, index=False)
size_csv_kb = os.path.getsize(OUTPUT_CSV) / 1024

print(f"\n{'='*50}")
print(f"SAVED:")
print(f"  HDF5 (primary) : {OUTPUT_HDF5} ({size_h5_kb:.1f} KB)")
print(f"  CSV (secondary): {OUTPUT_CSV} ({size_csv_kb:.1f} KB)")
print(f"{'='*50}")
print(f"  Total rows   : {len(df)}")
print(f"  Total graphs : {df['group_id'].nunique()}")
graph_sz = df.groupby("group_id").size()
print(f"  Min nodes    : {graph_sz.min()}")
print(f"  Max nodes    : {graph_sz.max()}")
print(f"  Avg nodes    : {graph_sz.mean():.1f}")
print(f"  Halo mass range (log10 M_sun): {df['halo_mass_log'].min():.2f} — "
      f"{df['halo_mass_log'].max():.2f}")
print(f"  Halo mass range (linear M_sun): {df['halo_mass'].min():.2e} — "
      f"{df['halo_mass'].max():.2e}")
print(f"\nGraph size distribution:")
for threshold in [3, 5, 10, 20, 50]:
    count = (graph_sz >= threshold).sum()
    print(f"  {threshold:3d}+ nodes: {count} halos")
print(f"\nSanity check (stellar_mass should be ~10-12, halo_mass_log should be ~12-15):")
print(df[["stellar_mass", "log_vel_dispersion",
          "log_half_mass_radius", "log_metallicity",
          "halo_mass_log"]].describe().round(3))