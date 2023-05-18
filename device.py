import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() == 'cuda':
    torch.backends.cudnn.benchmark = True
