import torch, yaml

# Load config FROM the checkpoint (this is the config the model was trained with)
ckpt = torch.load('outputs/checkpoints/best_model.pt', map_location='cpu')
trained_cfg = ckpt['config']

print('Config inside checkpoint:')
print(f"  hidden_dim : {trained_cfg['model']['hidden_dim']}")
print(f"  num_layers : {trained_cfg['model']['num_layers']}")
print(f"  dropout    : {trained_cfg['model']['dropout']}")
print(f"  pooling    : {trained_cfg['model']['pooling']}")
print(f"  radius_mpc : {trained_cfg['graph']['radius_mpc']}")
print(f"  epochs     : {trained_cfg['training']['epochs']}")

# Overwrite local config.yaml with the checkpoint config
# so local and checkpoint always match
trained_cfg['data']['source'] = 'tng'  # ensure tng source
with open('config/config.yaml', 'w') as f:
    yaml.dump(trained_cfg, f, default_flow_style=False)

print('\nconfig.yaml overwritten with checkpoint config')
print('Now run: python main.py explain')
