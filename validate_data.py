import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/tng100_clustered.csv')
print('=== DATA VALIDATION ===')
print(f'Total rows: {len(df)}')
print(f'Total graphs: {df["group_id"].nunique()}')
print(f'Columns: {list(df.columns)}')
print()

required = ['stellar_mass', 'vel_dispersion', 'half_mass_radius',
            'metallicity', 'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z', 'halo_mass_log', 'group_id']
missing = [c for c in required if c not in df.columns]
print(f'Missing columns: {missing if missing else "NONE - all good"}')
print()

nan_counts = df[required].isnull().sum()
print('NaN counts per column:')
has_nan = nan_counts[nan_counts > 0]
print(has_nan if len(has_nan) > 0 else '  None - all clean')
print()

sizes = df.groupby('group_id').size()
print(f'Graph sizes: min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}')
print()
print(f'Target range: {df["halo_mass_log"].min():.2f} to {df["halo_mass_log"].max():.2f}')
print()
print('=== VALIDATION COMPLETE ===')
