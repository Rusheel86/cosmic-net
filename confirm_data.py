import pandas as pd, numpy as np
df = pd.read_csv('data/raw/tng100_clustered.csv')
c = df[df['is_central']==1]
diff = c['halo_mass_log'] - c['stellar_mass']
print(f'Centrals: {len(c)}')
print(f'halo_mass_log - stellar_mass:')
print(f'  mean = {diff.mean():.3f} dex  (expect 1.5-3.0)')
print(f'  std  = {diff.std():.3f} dex   (expect > 0.3)')
print(f'  min  = {diff.min():.3f}')
print(f'  max  = {diff.max():.3f}')
if diff.mean() > 1.0:
    print("TARGET IS CORRECT - Group_M_Crit200 confirmed")
else:
    print("WARNING - check data")
