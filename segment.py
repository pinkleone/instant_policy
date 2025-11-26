from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np
import clip
import fastsam.prompt
fastsam.prompt.clip = clip

# Load fast sam model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FastSAM('./FastSAM-x.pt')

# Get text prompt based on task
PROMPTS = {
    'plate_out': 'round white dinner plate',
    'basketball': 'basketball',
    'toilet_seat_down': 'toilet',
    'toilet_seat_up': 'toilet',
}

def get_mask(image, task_name, bbox=None):
    # Perform segmentation
    results = model(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, verbose=False)
    
    fastsam_prompt = FastSAMPrompt(image, results, device=DEVICE)

    text_prompt = PROMPTS.get(task_name, 'object')

    if bbox is not None:
        # Use box prompt and text prompt together
        prompt_results = fastsam_prompt.box_prompt(bboxes=[bbox], text=text_prompt)
    else:
        prompt_results = fastsam_prompt.text_prompt(text=text_prompt)
    
    # If no results, return empty mask
    if len(prompt_results) == 0:
        print('No segmentation found.')
        return np.zeros(image.shape[:2])
    
    return np.any(prompt_results, axis=0)


