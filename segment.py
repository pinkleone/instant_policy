from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np

# Load fast sam model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FastSAM('./FastSAM-x.pt')

# Get text prompt based on task
PROMPTS = {
    'plate_out': 'plate'
}

def get_mask(image, task_name):
    text_prompt = PROMPTS.get(task_name, 'object')

    # Perform segmentation
    results = model(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, verbose=False)
    
    fastsam_prompt = FastSAMPrompt(image, results, device=DEVICE)

    prompt_results = fastsam_prompt.text_prompt(text=text_prompt)
    
    # If no results, return empty mask
    if len(prompt_results) == 0:
        return np.zeros(image.shape[:2])
    
    return np.any(prompt_results, axis=0)


