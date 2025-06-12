import argparse
import numpy as np
import tifffile
import cv2
from skimage import measure
from pathlib import Path
from tqdm import tqdm

def instance2semantics(ins):
    h, w = ins.shape[:2]
    tmp0 = ins[1:, 1:] - ins[:h - 1, :w - 1]
    ind0 = np.where(tmp0 != 0)

    tmp1 = ins[1:, :w - 1] - ins[:h - 1, 1:]
    ind1 = np.where(tmp1 != 0)
    ins[ind1] = 0
    ins[ind0] = 0
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)

def process_folder_i2s(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tif_paths = list(input_folder.glob("*.tif"))
    for tif_path in tqdm(tif_paths, desc="Instance to Semantic", unit="image"):
        try:
            mask = tifffile.imread(tif_path)
            mask = np.squeeze(mask)
            semantics = instance2semantics(mask)
            semantics[semantics > 0] = 255
            tifffile.imwrite(output_folder / tif_path.name, semantics, compression='zlib')
        except Exception as e:
            print(f"Failed to process {tif_path.name}: {e}")

def semantic2instance(mask_path, output_path):
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        labels = measure.label(mask, connectivity=1)
        unique_labels = np.unique(labels)

        label_assignments = np.zeros_like(labels, dtype=np.uint16)
        for i, label in enumerate(unique_labels):
            if label != 0:
                label_assignments[labels == label] = i + 1

        filename = mask_path.stem + '.tif'
        output_file = output_path / filename
        tifffile.imwrite(output_file, label_assignments, compression='zlib')
    except Exception as e:
        print(f"Failed to convert {mask_path.name}: {e}")

def process_folder_s2i(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tif_paths = list(input_folder.glob("*.tif"))
    for tif_path in tqdm(tif_paths, desc="Semantic to Instance", unit="image"):
        semantic2instance(tif_path, output_folder)

def main():
    parser = argparse.ArgumentParser(description="Convert between instance and semantic segmentation masks.")
    parser.add_argument('-i', '--input_folder', required=True, help="Input folder containing TIFF images.")
    parser.add_argument('-o', '--output_folder', required=True, help="Output folder to save converted images.")
    parser.add_argument('-m', '--mode', choices=['i2s', 's2i'], required=True, help="Conversion mode: i2s (instance to semantic) or s2i (semantic to instance).")
    
    args = parser.parse_args()

    print(f"Starting conversion mode: {args.mode.upper()}")

    if args.mode == 'i2s':
        process_folder_i2s(args.input_folder, args.output_folder)
    elif args.mode == 's2i':
        process_folder_s2i(args.input_folder, args.output_folder)

    print(f"Finished processing. Output saved to {args.output_folder}")

if __name__ == "__main__":
    main()