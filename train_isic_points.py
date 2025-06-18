# file: train_isic_points.py
import torch
import os
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import time
from PIL import Image
import torchvision.transforms as transforms

from model import ConditionalDiffusionModel224
from diffusion import DiscreteDiffusion
from dataset_isic_points import ISICPointDataset

class ValidationDataset(torch.utils.data.Dataset):
    """
    Dataset for validation. Loads the validation image and its corresponding
    FULL ground truth mask for calculating segmentation metrics.
    """
    def __init__(self, img_dir, full_mask_dir, file_list, img_size, img_ext, mask_suffix):
        self.img_dir = img_dir
        self.full_mask_dir = full_mask_dir
        self.files = file_list
        self.img_ext = img_ext
        self.mask_suffix = mask_suffix
        
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        base_name = img_name.replace(self.img_ext, '')
        
        img_path = os.path.join(self.img_dir, img_name)
        full_mask_path = os.path.join(self.full_mask_dir, f"{base_name}{self.mask_suffix}")

        cond_image = Image.open(img_path).convert('RGB')
        full_mask = Image.open(full_mask_path).convert('L')

        cond_tensor = self.img_transform(cond_image)
        full_mask_tensor = (self.mask_transform(full_mask) > 0.5).float()
        
        ### MODIFICATION START ###
        # Return the image name for saving visualizations
        return cond_tensor, full_mask_tensor, img_name
        ### MODIFICATION END ###

def validate_and_log_metrics(val_loader, model, diffusion, device, epoch, vis_dir, num_vis):
    model.eval()
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    vis_saved_count = 0

    ### MODIFICATION START ###
    # Define denormalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def denormalize(tensor):
        return tensor * std + mean
    ### MODIFICATION END ###

    with torch.no_grad():
        ### MODIFICATION START ###
        # Update loop to get image names
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=False)
        for cond_images, true_full_masks, img_names in pbar:
        ### MODIFICATION END ###
            cond_images = cond_images.to(device)
            true_full_masks = true_full_masks.to(device) # Move masks to device for calculations
            
            predicted_masks = diffusion.sample(model, cond_images.shape[-1], cond_images.shape[0], 1, cond_images)
            
            # Calculate TP, FP, TN, FN for the batch
            # Ensure masks are on the same device
            tp = (predicted_masks * true_full_masks).sum()
            fp = (predicted_masks * (1 - true_full_masks)).sum()
            fn = ((1 - predicted_masks) * true_full_masks).sum()
            tn = ((1 - predicted_masks) * (1 - true_full_masks)).sum()
            
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

            ### MODIFICATION START ###
            # Save visualization for a few samples
            if vis_saved_count < num_vis:
                # Denormalize input images for proper viewing
                denorm_cond_images = denormalize(cond_images).clamp(0, 1)

                # Loop through the batch to save individual images
                for i in range(len(cond_images)):
                    if vis_saved_count < num_vis:
                        # Prepare masks for grid view (convert to 3 channels)
                        pred_mask_rgb = predicted_masks[i].cpu().repeat(3, 1, 1)
                        true_mask_rgb = true_full_masks[i].cpu().repeat(3, 1, 1)
                        
                        # Create a grid: [Input Image, Predicted Mask, Ground Truth]
                        grid = make_grid([denorm_cond_images[i].cpu(), pred_mask_rgb, true_mask_rgb], nrow=3)
                        
                        save_path = os.path.join(vis_dir, f"epoch_{epoch}_sample_{img_names[i]}")
                        save_image(grid, save_path)
                        vis_saved_count += 1
            ### MODIFICATION END ###
    
    # Calculate metrics
    epsilon = 1e-6
    precision = total_tp / (total_tp + total_fp + epsilon)
    fpr = total_fp / (total_fp + total_tn + epsilon)
    
    print(f"Validation Metrics -> Precision: {precision.item():.4f} | FPR: {fpr.item():.4f}")
    model.train()
    
    ### MODIFICATION START ###
    # Return precision to track the best model
    return precision.item()
    ### MODIFICATION END ###

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    sample_dir = os.path.join(args.save_dir, "samples")
    ### MODIFICATION START ###
    vis_dir = os.path.join(args.save_dir, "validation_visuals")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    ### MODIFICATION END ###


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    # Training data (Images and POINT masks)
    train_files = sorted([f for f in os.listdir(args.train_img_dir) if f.lower().endswith(args.img_ext)])
    train_dataset = ISICPointDataset(
        img_dir=args.train_img_dir, mask_dir=args.train_mask_dir, file_list=train_files,
        image_size=args.img_size, augment=True, img_ext=args.img_ext, mask_suffix=args.mask_suffix
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Validation data (Images and FULL masks for metric calculation)
    val_files = sorted([f for f in os.listdir(args.val_img_dir) if f.lower().endswith(args.img_ext)])
    val_dataset = ValidationDataset(
        img_dir=args.val_img_dir, full_mask_dir=args.val_full_mask_dir, file_list=val_files,
        img_size=args.img_size, img_ext=args.img_ext, mask_suffix=args.mask_suffix
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Training on {len(train_files)} images, Validating on {len(val_files)} images.")

    # --- Model, Diffusion, Optimizer ---
    model = ConditionalDiffusionModel224(vgg_pretrained=True).to(device)
    diffusion = DiscreteDiffusion(timesteps=args.timesteps, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    ### MODIFICATION START ###
    best_precision = 0.0
    ### MODIFICATION END ###

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (condition_image, x_start) in enumerate(pbar):
            optimizer.zero_grad()
            condition_image, x_start = condition_image.to(device), x_start.to(device)
            loss = diffusion.compute_loss(model, x_start, condition_image)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # --- Validation and Saving ---
        if (epoch + 1) % args.save_interval == 0:
            print(f"\n--- Epoch {epoch+1} Summary ---")
            ### MODIFICATION START ###
            # Get precision from validation and save visuals
            current_precision = validate_and_log_metrics(
                val_loader, model, diffusion, device, epoch + 1, vis_dir, args.num_val_vis
            )
            
            # Save the latest checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            # Save the best model based on precision
            if current_precision > best_precision:
                best_precision = current_precision
                best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"*** New best model saved with precision: {best_precision:.4f} to {best_ckpt_path} ***")
            ### MODIFICATION END ###

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a diffusion model with separate validation.")
    # --- Paths ---
    parser.add_argument('--train_img_dir', type=str, required=True, help='Path to the PROCESSED training images.')
    parser.add_argument('--train_mask_dir', type=str, required=True, help='Path to the PROCESSED training POINT masks.')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Path to the ORIGINAL validation images.')
    parser.add_argument('--val_full_mask_dir', type=str, required=True, help='Path to the ORIGINAL validation FULL masks.')
    parser.add_argument('--save_dir', type=str, default='results_isic_points_validated', help='Directory to save results.')
    
    # --- File Naming ---
    parser.add_argument('--img_ext', type=str, default='.jpg', help='Image file extension.')
    parser.add_argument('--mask_suffix', type=str, default='_segmentation.png', help='Suffix for mask filenames.')

    # --- Training Hyperparameters ---
    parser.add_argument('--img_size', type=int, default=224, help='Image size.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--save_interval', type=int, default=2, help='Validate and save checkpoint every N epochs.')
    
    ### MODIFICATION START ###
    parser.add_argument('--num_val_vis', type=int, default=4, help='Number of validation samples to visualize.')
    ### MODIFICATION END ###
    
    # --- Diffusion Hyperparameters ---
    parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion timesteps.')
    
    args = parser.parse_args()
    train(args)

    #python train_isic_points.py --train_img_dir C:\Users\Mehmet_Postdoc\Desktop\ISIC_2017_Dataset\processed_points_5\train_images --train_mask_dir C:\Users\Mehmet_Postdoc\Desktop\ISIC_2017_Dataset\processed_points_5\train_point_masks --val_img_dir C:\Users\Mehmet_Postdoc\Desktop\ISIC_2017_Dataset\ISIC-2017_Validation_Data --val_full_mask_dir C:\Users\Mehmet_Postdoc\Desktop\ISIC_2017_Dataset\ISIC-2017_Validation_Part1_GroundTruth