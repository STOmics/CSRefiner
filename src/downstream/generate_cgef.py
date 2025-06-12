import os
import argparse
from typing import Optional
import logging
from gefpy import cgef_writer_cy
from gefpy.bgef_writer_cy import generate_bgef
from utils.utils import cbimread, cbimwrite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def gem_to_gef(gem_path, gef_path):
    generate_bgef(input_file=gem_path,
                  bgef_file=gef_path,
                  stromics="Transcriptomics",
                  n_thread=8,
                  bin_sizes=[1],
                  )

def adjust_mask_shape(gef_path, mask_path):
    m_width, m_height = cMatrix.gef_gef_shape(gef_path)
    mask = cbimread(mask_path)
    if mask.width == m_width and mask.height == m_height:
        return mask_path
    mask_adjust = mask.trans_image(offset=[0, 0], dst_size=(m_height, m_width))
    path_no_ext, ext = os.path.splitext(mask_path)
    new_path = path_no_ext + "_adjust" + ".tif"
    cbimwrite(new_path, mask_adjust)
    return new_path

def generate_cellbin(input_path: str, output_path: str, mask_path: str, block_size: Optional[list] = None) -> int:
    
    if block_size is None:
        block_size = [256, 256]
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return 1
    if not os.path.exists(mask_path):
        logger.error(f"Mask file not found: {mask_path}")
        return 1
    
    if input_path.endswith(".gem.gz"):
        gef_path = os.path.join(
            os.path.dirname(output_path),
            os.path.basename(input_path).replace(".gem.gz", ".raw.gef")
        )
        
        if os.path.exists(gef_path):
            logger.info(f"Using existing GEF file: {gef_path}")
            input_path = gef_path
        else:
            try:
                gem_to_gef(input_path, gef_path)
                input_path = gef_path
            except Exception as e:
                logger.error(f"Failed to convert GEM to GEF: {str(e)}")
                return 1
    
    if input_path.endswith(".gef"):
        try:
            adjusted_mask = adjust_mask_shape(input_path, mask_path)

            logger.info(f"Generating CGEF file: {output_path}")
            cgef_writer_cy.generate_cgef(output_path, input_path, adjusted_mask, block_size)
            logger.info("CGEF generation completed successfully")
        except Exception as e:
            logger.error(f"Failed to generate CGEF: {str(e)}")
            return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate cellbin matrix (CGEF) from expression matrix and cell segmentation mask")
    parser.add_argument("-i", "--input", required=True, help="Input file path (.gem.gz or .gef)")
    parser.add_argument("-o", "--output", required=True, help="Output CGEF file path")
    parser.add_argument("-m", "--mask", required=True, help="Cell segmentation mask file path")
    parser.add_argument("-b", "--block-size", type=int, nargs=2, default=[256, 256], help="Block size for CGEF generation (default: 256 256)")
    
    args = parser.parse_args()
    
    ret = generate_cellbin(args.input, args.output, args.mask, args.block_size)
    
    if ret != 0:
        logger.error("Failed to generate cellbin matrix")
    else:
        logger.info("Successfully generated cellbin matrix")
    
    exit(ret)

if __name__ == "__main__":
    main()