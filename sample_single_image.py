# file: sample_single_image.py
import torch
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from scipy.ndimage import binary_fill_holes # <-- Import the hole filling function

from model import ConditionalDiffusionModel224
from diffusion import DiscreteDiffusion

def sample_single(args, number_of_samples=100):
    """
    Generates a segmentation mask for a single input image using a trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Model and Diffusion ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = ConditionalDiffusionModel224(vgg_pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    diffusion = DiscreteDiffusion(timesteps=args.timesteps, num_classes=2).to(device)

    # --- 2. Prepare Input Image ---
    transform_img = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Processing input image: {args.input_img}")
    input_image = Image.open(args.input_img).convert("RGB")
    cond_tensor = transform_img(input_image).unsqueeze(0).to(device)
    
    output_masks = []
    # --- 3. Generate Predictions ---
    with torch.no_grad():
        for i in range(number_of_samples):
            # --- (MODIFICATION) CHOOSE SAMPLING METHOD ---
            if args.ddim:
                print(f"Generating segmentation mask using DDIM with {args.ddim_steps} steps...")
                predicted_mask = diffusion.sample_ddim(
                    model,
                    image_size=args.img_size,
                    batch_size=1,
                    condition_image=cond_tensor,
                    num_inference_steps=args.ddim_steps
                ).cpu()
            else:
                print(f"Generating segmentation mask using DDPM with {args.timesteps} steps...")
                predicted_mask = diffusion.sample(
                    model,
                    image_size=args.img_size,
                    batch_size=1,
                    condition_image=cond_tensor
                ).cpu()
            # --- (END MODIFICATION) ---
            output_masks.append(predicted_mask)

    # --- 4. Combine Masks ---
    all_masks_tensor = torch.cat(output_masks, dim=0) 
    combined_mask = torch.any(all_masks_tensor, dim=0, keepdim=True).float()

    # --- 4a. Apply Morphological Hole Filling ---
    print("Applying morphological hole filling...")
    # Convert the PyTorch tensor to a NumPy array for scipy
    # Squeeze to remove batch and channel dims -> [H, W], convert to boolean
    mask_np = combined_mask.squeeze().cpu().numpy().astype(bool)
    
    # Apply the hole filling operation
    filled_mask_np = binary_fill_holes(mask_np)
    
    # Convert the filled NumPy array back to a PyTorch tensor
    # Add batch and channel dims back -> [1, 1, H, W]
    filled_mask_tensor = torch.from_numpy(filled_mask_np).float().unsqueeze(0).unsqueeze(0)


    # --- 4b. Create Visualization Grid ---
    images_to_show = []

    # Un-normalize the input image for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    input_image_vis = (cond_tensor.cpu() * std + mean).clamp(0, 1)
    images_to_show.append(input_image_vis)

    # If ground truth mask is provided, add it to the grid
    if args.gt_mask:
        print(f"Loading ground truth mask: {args.gt_mask}")
        transform_mask = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        gt_mask_pil = Image.open(args.gt_mask).convert('L')
        gt_mask_tensor = (transform_mask(gt_mask_pil) > 0.5).float()
        images_to_show.append(gt_mask_tensor.unsqueeze(0).repeat(1, 3, 1, 1))

    # Add the original predicted mask
    images_to_show.append(combined_mask.repeat(1, 3, 1, 1))
    
    # Add the filled predicted mask
    images_to_show.append(filled_mask_tensor.repeat(1, 3, 1, 1))

    # Concatenate images horizontally: [Input | GT (Opt) | Prediction | Filled Prediction]
    comparison_grid = torch.cat(images_to_show, dim=3)

    # --- 5. Save Result ---
    os.makedirs(os.path.dirname(args.output_img), exist_ok=True)
    save_image(comparison_grid, args.output_img, normalize=False)
    print(f"\nSaved visualization grid to: {args.output_img}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a segmentation mask for a single ISIC image.")
    
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--input_img', type=str, required=True, help="Path to the input image file.")
    parser.add_argument('--output_img', type=str, required=True, help="Path to save the output visualization grid.")
    parser.add_argument('--gt_mask', type=str, default=None, help="(Optional) Path to the ground truth full mask for comparison.")
    parser.add_argument('--img_size', type=int, default=224, help="Image size used during training.")
    parser.add_argument('--timesteps', type=int, default=200, help="Number of diffusion timesteps used during training.")
    
    # --- (NEW) DDIM ARGUMENTS ---
    parser.add_argument('--ddim', action='store_true', help="Use DDIM for faster sampling instead of DDPM.")
    parser.add_argument('--ddim_steps', type=int, default=50, help="Number of inference steps for DDIM. Only used if --ddim is set.")
    # --- (END NEW) ---
    
    args = parser.parse_args()
    sample_single(args)
    #python sample_single_image.py --ddim --ddim_steps 50 --checkpoint C:\Users\Mehmet_Postdoc\Desktop\python_set_up_code\Diffusion_model_for_ISIC_Segmentation\results_isic_points_validated\checkpoints\best_model.pth --input_img C:\Users\Mehmet_Postdoc\Desktop\ISIC_2017_Dataset\ISIC-2017_Test_v2_Data\ISIC_0012092.jpg --output_img C:\Users\Mehmet_Postdoc\Desktop\python_set_up_code\Diffusion_model_for_ISIC_Segmentation\results_isic_points_validated\samples\output_image.png --gt_mask C:\Users\Mehmet_Postdoc\Desktop\ISIC_2017_Dataset\ISIC-2017_Test_v2_Part1_GroundTruth\ISIC_0012092_segmentation.png