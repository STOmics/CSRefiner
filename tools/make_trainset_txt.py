import os
import glob
import argparse

def generate_file_pairs(img_dir, mask_dir, output_txt):
    """generate the path of the image and mask files and write them to the specified txt file"""
    matched_pairs = []

    img_files = glob.glob(os.path.join(img_dir, "*-img.tif"))

    for img_file in img_files:
        filename = os.path.basename(img_file)
        mask_file = os.path.join(mask_dir, filename.replace("-img", "-mask"))

        if not os.path.exists(img_file) or not os.path.exists(mask_file):
            print(f"[Warning] Missing file for: {filename}")
            continue

        matched_pairs.append(f"{img_file},{mask_file}")

    with open(output_txt, mode="a", encoding="utf-8") as f:
        f.write("\n".join(matched_pairs) + "\n")

    print(f"[Info] Wrote {len(matched_pairs)} entries to {output_txt}")

def main():
    parser = argparse.ArgumentParser(description="Generate image-mask file list.")
    parser.add_argument("-img", "--img_dir", required=True, help="Path to image directory.")
    parser.add_argument("-mask", "--mask_dir", required=True, help="Path to mask directory.")
    parser.add_argument("-o", "--output_txt", required=True, help="Output txt file path.")
    args = parser.parse_args()

    generate_file_pairs(args.img_dir, args.mask_dir, args.output_txt)


if __name__ == "__main__":
    main()