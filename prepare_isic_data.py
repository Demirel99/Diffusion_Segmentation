# file: prepare_isic_data.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def generate_and_save_point_masks(
    input_img_dir, input_full_mask_dir, output_img_dir, output_point_mask_dir,
    mask_suffix='_segmentation.png', num_points=10, image_size=224
):
    """
    Processes the ISIC dataset to generate sparse point masks for weakly-supervised training.

    For each image, it:
    1. Resizes and copies the original image.
    2. Samples `num_points` from the corresponding full segmentation mask.
    3. Saves this new sparse point mask.
    """
    print("Starting data preprocessing for point supervision...")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_point_mask_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_img_dir}")
    print(f"Output point masks will be saved to: {output_point_mask_dir}")

    image_files = sorted([f for f in os.listdir(input_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    transform_mask = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    transform_image = transforms.Resize((image_size, image_size))

    for img_name in tqdm(image_files, desc="Processing Images"):
        base_name = os.path.splitext(img_name)[0]
        full_mask_name = f"{base_name}{mask_suffix}"

        input_img_path = os.path.join(input_img_dir, img_name)
        input_full_mask_path = os.path.join(input_full_mask_dir, full_mask_name)

        if not os.path.exists(input_full_mask_path):
            print(f"Warning: Mask not found for {img_name}, skipping.")
            continue

        full_mask = Image.open(input_full_mask_path).convert("L")
        full_mask_tensor = transform_mask(full_mask)

        # Find coordinates of the lesion (foreground)
        foreground_coords = torch.nonzero(full_mask_tensor > 0.5)
        point_mask_tensor = torch.zeros_like(full_mask_tensor)

        if len(foreground_coords) > 0:
            foreground_coords = foreground_coords[:, 1:] # Drop the channel dimension
            num_foreground_pixels = foreground_coords.shape[0]
            
            # Sample with replacement if not enough pixels
            sample_count = min(num_points, num_foreground_pixels)
            if sample_count > 0:
                random_indices = torch.randint(0, num_foreground_pixels, (sample_count,))
                sampled_points = foreground_coords[random_indices]
                point_mask_tensor[0, sampled_points[:, 0], sampled_points[:, 1]] = 1.0

        # --- Save the results ---
        # a) Resize and save the original image
        img = Image.open(input_img_path).convert("RGB")
        img_resized = transform_image(img)
        output_img_path = os.path.join(output_img_dir, img_name)
        img_resized.save(output_img_path)

        # b) Save the point mask (using the original mask name for simplicity)
        point_mask_np = point_mask_tensor.squeeze(0).mul(255).byte().cpu().numpy()
        point_mask_image = Image.fromarray(point_mask_np, mode='L')
        output_mask_path = os.path.join(output_point_mask_dir, full_mask_name)
        point_mask_image.save(output_mask_path)

    print(f"\nPreprocessing complete! Processed {len(image_files)} images.")


if __name__ == '__main__':
    # --- USER-DEFINED PATHS ---
    # This script should be run for your training set.
    ROOT_DATA_PATH = r"C:\Users\Mehmet_Postdoc\Desktop\ISIC_2017_Dataset" # CHANGE THIS
    
    # 1. DEFINE YOUR **INPUT** DIRECTORIES (Original ISIC Data)
    INPUT_IMG_DIR = os.path.join(ROOT_DATA_PATH, "ISIC-2017_Training_Data")
    INPUT_FULL_MASK_DIR = os.path.join(ROOT_DATA_PATH, "ISIC-2017_Training_Part1_GroundTruth")

    # 2. DEFINE YOUR **OUTPUT** DIRECTORIES (Processed Data for Training)
    OUTPUT_ROOT = os.path.join(ROOT_DATA_PATH, f"processed_points_{2}")
    OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "train_images")
    OUTPUT_POINT_MASK_DIR = os.path.join(OUTPUT_ROOT, "train_point_masks")

    # 3. DEFINE PARAMETERS
    NUM_POINTS = 2
    IMAGE_SIZE = 224
    MASK_SUFFIX = "_segmentation.png"

    generate_and_save_point_masks(
        input_img_dir=INPUT_IMG_DIR,
        input_full_mask_dir=INPUT_FULL_MASK_DIR,
        output_img_dir=OUTPUT_IMG_DIR,
        output_point_mask_dir=OUTPUT_POINT_MASK_DIR,
        mask_suffix=MASK_SUFFIX,
        num_points=NUM_POINTS,
        image_size=IMAGE_SIZE
    )