import torch, os

checkpoint = torch.load('GNN_final_model.pt', map_location='cpu')

torch.save({
    'model_state_dict': checkpoint['model_state_dict'],
    'config':           checkpoint['config'],
    'metrics':          checkpoint['metrics'],
    'epoch':            checkpoint['epoch'],
}, 'outputs/checkpoints/best_model.pt')

original = os.path.getsize('GNN_final_model.pt') / 1024 / 1024
fixed    = os.path.getsize('outputs/checkpoints/best_model.pt') / 1024 / 1024
print(f'Original : {original:.1f} MB')
print(f'Fixed    : {fixed:.1f} MB')
print(f'Epoch    : {checkpoint["epoch"]}')
print('Metrics:')
for k, v in checkpoint['metrics'].items():
    print(f'  {k}: {v}')
