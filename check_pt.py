import torch

checkpoint = torch.load('GNN_final_model.pt', map_location='cpu')

# See what keys are stored
if isinstance(checkpoint, dict):
    print("Keys in checkpoint:")
    for k, v in checkpoint.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        else:
            print(f"  {k}: {type(v)}")
else:
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"Size check: not a dict")