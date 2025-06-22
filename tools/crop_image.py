import os
import re
import tifffile
import numpy as np
from tqdm import tqdm
import argparse

def parse_filename(filename):
    """
    Parse filename to extract chip_id and suffix (img/mask).
    If filename already has coordinates, extract chip_id and base_x, base_y.
    """
    match = re.match(
        r'(?P<chip_id>.+)-x(?P<x>\d+)_y(?P<y>\d+)_w(?P<w>\d+)_h(?P<h>\d+)-(?P<suffix>img|mask)\.tif$',
        filename
    )
    if match:
        return (
            match.group('chip_id'),
            int(match.group('x')),
            int(match.group('y')),
            int(match.group('w')),
            int(match.group('h')),
            match.group('suffix')
        )
    else:
        if filename.endswith('-img.tif'):
            suffix = 'img'
            chip_id = filename[:-9]  
        elif filename.endswith('-mask.tif'):
            suffix = 'mask'
            chip_id = filename[:-10] 
        else:
            suffix = 'img'
            chip_id = os.path.splitext(filename)[0]
        return (chip_id, 0, 0, None, None, suffix)

def crop_image(image, crop_size):
    crops = []
    H, W = image.shape[:2]
    step_y, step_x = crop_size
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            crop = image[y:y+step_y, x:x+step_x]
            if crop.shape[0] == step_y and crop.shape[1] == step_x:
                crops.append((crop, x, y))
    return crops

def process_file(file_path, output_folder, crop_size=(256, 256)):
    file_name = os.path.basename(file_path)
    image = tifffile.imread(file_path)
    image = np.squeeze(image)

    chip_id, base_x, base_y, _, _, suffix = parse_filename(file_name)
    crops = crop_image(image, crop_size)

    for crop, x, y in crops:
        new_x = base_x + x
        new_y = base_y + y
        out_name = f"{chip_id}-x{new_x}_y{new_y}_w{crop_size[0]}_h{crop_size[1]}-{suffix}.tif"
        tifffile.imwrite(os.path.join(output_folder, out_name), crop, compression='zlib')

def process_input(input_path, output_folder, crop_size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)

    if os.path.isfile(input_path) and input_path.endswith('.tif'):
        tqdm.write(f"Processing file: {input_path}")
        process_file(input_path, output_folder, crop_size)
    elif os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith('.tif')]
        for file_name in tqdm(files, desc="Cropping"):
            input_path_file = os.path.join(input_path, file_name)
            process_file(input_path_file, output_folder, crop_size)
    else:
        raise ValueError("Input is not a valid .tif file or directory.")

def main():
    parser = argparse.ArgumentParser(description="Crop large chip images and masks into patches.")
    parser.add_argument('-i', '--input', required=True, help="Input path: a .tif file or folder containing .tif files.")
    parser.add_argument('-o', '--output', required=True, help="Folder to save cropped patches.")
    parser.add_argument('-s', '--size', type=int, default=256, help="Patch size (default: 256)")

    args = parser.parse_args()
    process_input(args.input, args.output, (args.size, args.size))

if __name__ == "__main__":
    main()