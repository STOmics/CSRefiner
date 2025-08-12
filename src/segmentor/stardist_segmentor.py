import argparse
import os
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.io import imread, imsave
import glob
import tifffile
from tqdm import tqdm

# Supported official model names
OFFICIAL_MODELS = ['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018', '2D_demo']

def load_model(model_path):
    if model_path in OFFICIAL_MODELS:
        print(f"Loading official model: {model_path}")
        model = StarDist2D.from_pretrained(model_path)
    else:
        print(f"Loading local model weights: {model_path}")
        # Initialize with any official model, then load weights
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        model.keras_model.load_weights(model_path)
    return model

def process_image(model, input_path, output_path):
    img = imread(input_path)
    img_norm = normalize(img, 1, 99.8)
    labels, details = model.predict_instances(img_norm)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, labels.astype('uint16'), compression='zlib')
    print(f"Segmentation result saved to: {output_path}")

def generate_output_name(input_path):
    filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(filename)[0]
    pred_name = f"{name_without_ext}_stardist_mask.tif"
    return pred_name

def main():
    parser = argparse.ArgumentParser(description="StarDist cell segmentation inference script (supports official and fine-tuned models)")
    parser.add_argument('-p', "--model_path", required=True, help="Model name (official model like 2D_versatile_fluo, or local .h5 weights file path)")
    parser.add_argument('-i', '--input', required=True, help="Input image path or folder")
    parser.add_argument('-o', '--output', required=True, help="Output image path (if input is image) or output folder (if input is folder)")
    args = parser.parse_args()

    model = load_model(args.model_path)

    if os.path.isdir(args.input):
        # Batch mode: process all images in folder
        exts = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')
        img_files = []
        for ext in exts:
            img_files.extend(glob.glob(os.path.join(args.input, ext)))
        if not img_files:
            print(f"No images found in folder: {args.input}")
            return
        print(f"Found {len(img_files)} images. Starting segmentation...")
        for img_path in tqdm(img_files, desc="Processing images"):
            pred_name = generate_output_name(img_path)
            out_path = os.path.join(args.output, pred_name)
            process_image(model, img_path, out_path)
    else:
        # Single image mode
        pred_name = generate_output_name(args.input)
        out_path = os.path.join(args.output, pred_name)
        process_image(model, args.input, out_path)

if __name__ == "__main__":
    main() 