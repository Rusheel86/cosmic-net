import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/tng100_clustered.csv')

halo_features = df.groupby('group_id').agg({
    'stellar_mass'    : 'sum',
    'vel_dispersion'  : 'mean',
    'half_mass_radius': 'mean',
    'metallicity'     : 'mean',
    'halo_mass_log'   : 'first'
}).reset_index()

log_sm = np.log10(halo_features['stellar_mass'].clip(lower=1e6))
print(f"log_stellar_mass stats:")
print(f"  min : {log_sm.min():.3f}")
print(f"  max : {log_sm.max():.3f}")
print(f"  std : {log_sm.std():.3f}")
print(f"  NaN : {log_sm.isna().sum()}")
print(f"  Inf : {np.isinf(log_sm).sum()}")

corr = np.corrcoef(log_sm, halo_features['halo_mass_log'].values)[0,1]
print(f"  r with M_halo: {corr:.4f}")
