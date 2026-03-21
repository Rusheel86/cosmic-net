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
import os
import time

# ── CONFIG ──────────────────────────────────────────────────
API_KEY       = "a74da8e879fb9df993b5df59e1eb8670"
BASE_URL      = "https://www.tng-project.org/api"
SIM           = "TNG100-1"
SNAPSHOT      = 99
N_HALOS       = 500
MIN_HALO_MASS = 12.5
MIN_SUBHALOS  = 3
MAX_SUBHALOS_PER_HALO = 10  # cap satellites per halo for balanced graphs
OUTPUT_CSV    = "data/raw/tng100_clustered.csv"
CHECKPOINT    = "data/raw/checkpoint.csv"
HEADERS       = {"api-key": API_KEY}

os.makedirs("data/raw", exist_ok=True)

# ── HELPER ──────────────────────────────────────────────────
def tng_get(url, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params,
                           headers=HEADERS, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None
            elif r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n  Rate limited — waiting {wait}s...")
                time.sleep(wait)
            elif r.status_code == 404:
                return None
            else:
                time.sleep(5)
        except Exception as e:
            time.sleep(5)
    return None

# ── STEP 1: Verify connection ────────────────────────────────
print("Testing connection...")
sim_info = tng_get(f"{BASE_URL}/{SIM}/")
if sim_info is None:
    raise RuntimeError("Cannot connect. Check API key.")
print(f"Connected: {sim_info['name']}")

# ── STEP 2: Pull top N central subhalos ─────────────────────
print(f"\nPulling top {N_HALOS} central subhalos...")
all_centrals = []
url = f"{BASE_URL}/{SIM}/snapshots/{SNAPSHOT}/subhalos/"
params = {
    "limit"        : 100,
    "order_by"     : "-mass_log_msun",
    "primary_flag" : 1,
}

while url and len(all_centrals) < N_HALOS:
    data = tng_get(url, params=params)
    if data is None or "results" not in data:
        break
    all_centrals.extend(data["results"])
    url    = data.get("next")
    params = None
    # Progress bar
    pct = len(all_centrals) / N_HALOS * 100
    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    print(f"\r  [{bar}] {len(all_centrals)}/{N_HALOS}", end="", flush=True)
    time.sleep(0.3)

print(f"\nPulled {len(all_centrals)} centrals")

# ── STEP 3: Fetch satellites per halo ───────────────────────
print("\nFetching satellites per halo...")
print("Progress saves every 50 halos — safe to Ctrl+C and resume\n")

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

for halo_idx, central in enumerate(all_centrals):

    # ETA calculation
    if halo_idx > 0 and len(completed_ids) > 0:
        elapsed   = time.time() - start_time
        per_halo  = elapsed / max(len(completed_ids), 1)
        remaining = (N_HALOS - len(completed_ids)) * per_halo
        eta_mins  = remaining / 60
        eta_str   = f"ETA: {eta_mins:.0f}m"
    else:
        eta_str = "ETA: calculating..."

    # Get central detail for group_id
    central_detail = tng_get(central["url"])
    if central_detail is None:
        skipped += 1
        continue

    group_id      = central_detail.get("grnr", None)
    halo_mass_log = central_detail.get("mass_log_msun", None)

    if group_id is None or halo_mass_log is None:
        skipped += 1
        continue

    if group_id in completed_ids:
        continue

    if float(halo_mass_log) < MIN_HALO_MASS:
        skipped += 1
        continue

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

            halo_rows.append({
                "subhalo_id"       : int(sub["id"]),
                "group_id"         : int(group_id),
                "is_central"       : int(d.get("primary_flag", 0) or 0),
                "stellar_mass"     : sf("mass_log_msun"),
                "vel_dispersion"   : sf("veldisp"),
                "half_mass_radius" : sf("halfmassrad"),
                "metallicity"      : sf("gasmetallicity"),
                "pos_x"            : sf("pos_x"),
                "pos_y"            : sf("pos_y"),
                "pos_z"            : sf("pos_z"),
                "vel_x"            : sf("vel_x"),
                "vel_y"            : sf("vel_y"),
                "vel_z"            : sf("vel_z"),
                "halo_mass_log"    : float(halo_mass_log),
            })
        except:
            continue
        time.sleep(0.03)

    if len(halo_rows) >= MIN_SUBHALOS:
        all_rows.extend(halo_rows)
        completed_ids.add(group_id)
    else:
        skipped += 1

    # Progress print every 10 halos
    if len(completed_ids) % 10 == 0 and len(completed_ids) > 0:
        graphs  = len(completed_ids)
        subs    = len(all_rows)
        avg_n   = subs / max(graphs, 1)
        pct     = graphs / N_HALOS * 100
        bar     = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {graphs}/{N_HALOS} halos | "
              f"{subs} subhalos | avg {avg_n:.1f} nodes | "
              f"{eta_str}     ", end="", flush=True)

    # Checkpoint every 50 halos
    if len(completed_ids) % 50 == 0 and len(completed_ids) > 0:
        pd.DataFrame(all_rows).to_csv(CHECKPOINT, index=False)
        print(f"\n  Checkpoint saved at {len(completed_ids)} halos")

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

# ── STEP 5: Save ─────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
size_kb = os.path.getsize(OUTPUT_CSV) / 1024

print(f"\n{'='*50}")
print(f"SAVED: {OUTPUT_CSV}")
print(f"{'='*50}")
print(f"  File size    : {size_kb:.1f} KB")
print(f"  Total rows   : {len(df)}")
print(f"  Total graphs : {df['group_id'].nunique()}")
graph_sz = df.groupby("group_id").size()
print(f"  Min nodes    : {graph_sz.min()}")
print(f"  Max nodes    : {graph_sz.max()}")
print(f"  Avg nodes    : {graph_sz.mean():.1f}")
print(f"  Mass range   : {df['halo_mass_log'].min():.2f} — "
      f"{df['halo_mass_log'].max():.2f} log10(Msun)")
print(f"\nGraph size distribution:")
for threshold in [3, 5, 10, 20, 50]:
    count = (graph_sz >= threshold).sum()
    print(f"  {threshold:3d}+ nodes: {count} halos")
print(f"\nSanity check:")
print(df[["stellar_mass", "log_vel_dispersion",
          "log_half_mass_radius", "log_metallicity",
          "halo_mass_log"]].describe().round(3))