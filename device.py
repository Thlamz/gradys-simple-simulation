import torch

device = "cpu"
if torch.cuda.is_available() == 'cuda':
    torch.backends.cudnn.benchmark = True
