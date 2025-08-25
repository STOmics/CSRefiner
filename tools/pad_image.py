import os
import cv2
import tifffile
import argparse

def pad_to_target(image, target_size):
    h, w = image.shape[:2]
    th, tw = target_size
    
    pad_h = th - h
    pad_w = tw - w
    
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target size must be larger than the input image size.")
    
    return cv2.copyMakeBorder(
        image,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        borderType=cv2.BORDER_REFLECT_101
    )

def batch_pad_images(images_dir, masks_dir, out_images_dir, out_masks_dir, target_size):
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith("-img.tif")])

    for img_file in image_files:
        mask_file = img_file.replace("-img.tif", "-mask.tif")
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_file}, skipping.")
            continue

        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        img_padded = pad_to_target(img, target_size)
        mask_padded = pad_to_target(mask, target_size)

        tifffile.imwrite(os.path.join(out_images_dir, img_file), img_padded)
        tifffile.imwrite(os.path.join(out_masks_dir, mask_file), mask_padded)

        print(f"Processed: {img_file}  Original size {img.shape} → New size {img_padded.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pad images and their corresponding masks to a specified target size.")
    parser.add_argument('-i', '--images_dir', required=True, help='Directory containing original images.')
    parser.add_argument('-g', '--masks_dir', required=True, help='Directory containing masks.')
    parser.add_argument('-io', '--out_images_dir', required=True, help='Output directory for padded images.')
    parser.add_argument('-go', '--out_masks_dir', required=True, help='Output directory for padded masks.')
    parser.add_argument('-ts', '--target_size', required=True, type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'), 
                        help='Target size (height width) for padding.')

    args = parser.parse_args()
    
    batch_pad_images(args.images_dir, args.masks_dir, args.out_images_dir, args.out_masks_dir, tuple(args.target_size))