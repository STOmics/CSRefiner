import sys
import os
sys.path.append(os.getcwd())

import glob
import argparse
from traceback import print_exc
import tifffile
from utils.utils import auto_make_dir
from cellbin.modules.cell_segmentation import CellSegmentation
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument('-p', "--model_path", help="model path")
    parser.add_argument("-g", "--gpu", help="the gpu index", default="-1")
    parser.add_argument("-th", "--num_threads", help="num_threads", default="0")

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    model_path = args.model_path
    gpu = args.gpu
    num_threads = args.num_threads
    os.makedirs(output_path, exist_ok=True)
    file_lst = []
    if os.path.isdir(input_path):
        file_lst = glob.glob(os.path.join(input_path, "*.tif")) + glob.glob(os.path.join(input_path, "*.png")) + glob.glob(os.path.join(input_path, "*.jpg"))
    else:
        file_lst = [input_path]

    cell_bcdu = CellSegmentation(
        model_path=model_path,
        gpu=gpu,
        num_threads=num_threads
    )

    for i,file in enumerate(tqdm(file_lst, desc='v3')):
        try:
            _, name = os.path.split(file)
            out_file = auto_make_dir(file, src=input_path, output=output_path)

            dirname = os.path.dirname(out_file)
            basename = os.path.splitext(os.path.basename(out_file))[0] + "_v3_mask" + os.path.splitext(out_file)[1]
            out_file = os.path.join(dirname, basename)

            if os.path.exists(out_file):
                continue
            img = tifffile.imread(file)
            mask = cell_bcdu.run(img)
            
            tifffile.imwrite(out_file, mask, compression='zlib')
        except:
            print_exc()


if __name__ == '__main__':
    import sys

    main()
    sys.exit()
