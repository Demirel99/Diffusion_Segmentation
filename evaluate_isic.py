# file: evaluate_isic.py
import torch
import os
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from model import ConditionalDiffusionModel224
from diffusion import DiscreteDiffusion

class TestDataset(torch.utils.data.Dataset):
    """
    Specialized dataset for evaluation. Loads the test image and the 
    corresponding FULL ground truth mask for metric calculation.
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
        
        return cond_tensor, full_mask_tensor


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model for evaluation...")
    model = ConditionalDiffusionModel224(vgg_pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    diffusion = DiscreteDiffusion(timesteps=args.timesteps, num_classes=2).to(device)

    # Use the ORIGINAL test images and FULL masks for evaluation
    test_files = sorted([f for f in os.listdir(args.test_img_dir) if f.lower().endswith(args.img_ext)])
    
    test_dataset = TestDataset(
        img_dir=args.test_img_dir, full_mask_dir=args.test_mask_dir,
        file_list=test_files, img_size=args.img_size,
        img_ext=args.img_ext, mask_suffix=args.mask_suffix
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    total_dice, total_jaccard = 0.0, 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating Segmentation")
        for i, (cond_images, true_full_masks) in enumerate(pbar):
            cond_images = cond_images.to(device)
            
            # Generate the predicted full mask from the condition image
            predicted_masks = diffusion.sample(model, args.img_size, cond_images.shape[0], 1, cond_images)
            
            # --- Calculate Metrics against the ground truth full mask ---
            predicted_masks = predicted_masks.cpu()
            epsilon = 1e-6
            intersection = (predicted_masks * true_full_masks).sum(dim=(1, 2, 3))
            
            dice_denom = predicted_masks.sum(dim=(1, 2, 3)) + true_full_masks.sum(dim=(1, 2, 3))
            dice_score = (2. * intersection + epsilon) / (dice_denom + epsilon)
            total_dice += dice_score.sum().item()

            union = dice_denom - intersection
            jaccard_score = (intersection + epsilon) / (union + epsilon)
            total_jaccard += jaccard_score.sum().item()
            
            # Save visual samples
            if i * args.batch_size < args.num_visuals:
                mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
                cond_img_vis = cond_images.cpu() * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
                true_mask_vis = true_full_masks.repeat(1, 3, 1, 1)
                pred_mask_vis = predicted_masks.repeat(1, 3, 1, 1)
                
                comparison_grid = torch.cat([cond_img_vis, true_mask_vis, pred_mask_vis], dim=3)
                out_path = os.path.join(args.save_dir, f"eval_sample_batch_{i}.png")
                save_image(comparison_grid, out_path, normalize=False, nrow=1)

    num_images = len(test_dataset)
    print(f"\n--- Evaluation Complete on {num_images} images ---")
    print(f"Average Dice Score: {total_dice / num_images:.4f}")
    print(f"Average Jaccard Index (IoU): {total_jaccard / num_images:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a weakly-supervised diffusion model for ISIC segmentation.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    # NOTE: Use original, unprocessed test data for evaluation
    parser.add_argument('--test_img_dir', type=str, required=True, help='Path to the ORIGINAL test images directory.')
    parser.add_argument('--test_mask_dir', type=str, required=True, help='Path to the ORIGINAL, FULL test masks directory.')
    parser.add_argument('--save_dir', type=str, default='eval_results_isic_points', help='Directory to save results.')
    parser.add_argument('--img_size', type=int, default=224, help='Image size.')
    parser.add_argument('--img_ext', type=str, default='.jpg', help='Image file extension.')
    parser.add_argument('--mask_suffix', type=str, default='_segmentation.png', help='Suffix for mask filenames.')
    parser.add_argument('--timesteps', type=int, default=200, help="Diffusion timesteps (must match training).")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--num_visuals', type=int, default=16, help="Number of sample images to save.")
    args = parser.parse_args()
    evaluate(args)