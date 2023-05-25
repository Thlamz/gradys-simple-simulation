import torch

device = "cpu"
torch.set_num_threads(1)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
