from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np

# Load fast sam model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FastSAM('./FastSAM-x.pt')