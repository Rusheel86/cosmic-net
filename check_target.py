import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/tng100_clustered.csv')

# Check if halo_mass_log == stellar_mass for centrals
centrals = df[df['is_central'] == 1].copy()
print(f"Central subhalos: {len(centrals)}")
print(f"\nComparing halo_mass_log vs stellar_mass for centrals:")
print(f"  halo_mass_log range: {centrals['halo_mass_log'].min():.3f} to {centrals['halo_mass_log'].max():.3f}")
print(f"  stellar_mass range : {centrals['stellar_mass'].min():.3f} to {centrals['stellar_mass'].max():.3f}")
diff = centrals['halo_mass_log'] - centrals['stellar_mass']
print(f"  halo_mass_log - stellar_mass: mean={diff.mean():.4f}, std={diff.std():.4f}")
print(f"\nIf std is near 0, halo_mass_log == stellar_mass (data bug)")
print(f"If mean is ~1.5-2.0 and std>0.1, halo_mass_log is real M_crit200")

# Check correlation between stellar_mass and halo_mass_log
r = np.corrcoef(centrals['stellar_mass'], centrals['halo_mass_log'])[0,1]
print(f"\nCorrelation: r = {r:.6f}")
print(f"If r > 0.9999: data leakage confirmed")
print(f"If r < 0.95: data is probably correct")
