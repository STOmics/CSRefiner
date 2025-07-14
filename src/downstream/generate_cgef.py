import os
import argparse
import h5py
import numpy as np
import tifffile
import tempfile
from pathlib import Path
from typing import Optional
import logging
from gefpy import cgef_writer_cy
from gefpy.bgef_writer_cy import generate_bgef
from utils.utils import cbimread, cbimwrite, instance2semantics
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneticStandards(BaseModel):
    bin20_thr: int = Field(-1, description="")
    bin50_thr: int = Field(-1, description="")
    bin200_thr: int = Field(-1, description="")


class cMatrix(object):
    """ single matrix management """

    def __init__(self) -> None:
        self._gene_mat = np.array([])
        self.x_start = 65535
        self.y_start = 65535
        self.h_x_start = 0
        self.h_y_start = 0

        self._template: TemplateInfo = None
        self._chip_box: ChipBoxInfo = None
        self.file_path: str = ''

    def read(self, file_path: Path, chunk_size=1024 * 1024 * 10):
        """
        this function copy from,
            https://dcscode.genomics.cn/stomics/saw/register/-/blob/main/register/utils/matrixloader.py?ref_type=heads
        :param file_path: matrix file path
        :param chunk_size:
        :return:
        """
        suffix = file_path.suffix
        assert suffix in ['.gz', '.gef', '.gem']
        if suffix == ".gef":
            self.x_start, self.y_start, self._gene_mat = self._load_gef(file_path)
            return

        img = np.zeros((1, 1), np.uint8)
        if suffix == ".gz":
            fh = gzip.open(file_path, "rb")
        else:
            fh = open(str(file_path), "rb")  # pylint: disable=consider-using-with
        title = ""
        # Move pointer to the header of line
        eoh = 0
        header = ""
        for line in fh:
            line = line.decode("utf-8")
            if not line.startswith("#"):
                title = line
                break
            header += line
            eoh = fh.tell()
        fh.seek(eoh)
        # Initlise
        title = title.strip("\n").split("\t")
        umi_count_name = [i for i in title if "ount" in i][0]
        title = ["x", "y", umi_count_name]
        # todo There is a problem reading gem.gz and barcode_gene_exp.txt
        df = pd.read_csv(
            fh,
            sep="\t",
            header=0,
            usecols=title,
            dtype=dict(zip(title, [np.uint32] * 3)),
            chunksize=chunk_size,
        )

        _list = header.split("\n#")[-2:]
        self.h_x_start = int(_list[0].split("=")[1])
        self.h_y_start = int(_list[1].split("=")[1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for chunk in df:
                # convert data to image
                tmp_h = chunk["y"].max() + 1
                tmp_w = chunk["x"].max() + 1
                tmp_min_y = chunk["y"].min()
                tmp_min_x = chunk["x"].min()
                if tmp_min_x < self.x_start:
                    self.x_start = tmp_min_x
                if tmp_min_y < self.y_start:
                    self.y_start = tmp_min_y

                h, w = img.shape[:2]

                chunk = (
                    chunk.groupby(["x", "y"])
                        .agg(UMI_sum=(umi_count_name, "sum"))
                        .reset_index()
                )
                chunk["UMI_sum"] = chunk["UMI_sum"].mask(chunk["UMI_sum"] > 255, 255)
                tmp_img = np.zeros(shape=(tmp_h, tmp_w), dtype=np.uint8)
                tmp_img[chunk["y"], chunk["x"]] = chunk["UMI_sum"]

                # resize matrix
                ext_w = tmp_w - w
                ext_h = tmp_h - h
                if ext_h > 0:
                    img = np.pad(img, ((0, abs(ext_h)), (0, 0)), "constant")
                elif ext_h < 0:
                    tmp_img = np.pad(tmp_img, ((0, abs(ext_h)), (0, 0)), "constant")
                if ext_w > 0:
                    img = np.pad(img, ((0, 0), (0, abs(ext_w))), "constant")
                elif ext_w < 0:
                    tmp_img = np.pad(tmp_img, ((0, 0), (0, abs(ext_w))), "constant")

                # incase overflow
                tmp_img = (
                        255 - tmp_img
                )  # old b is gone shortly after new array is created
                np.putmask(
                    img, tmp_img < img, tmp_img
                )  # a temp bool array here, then it's gone
                img += 255 - tmp_img  # a temp array here, then it's gone
        df.close()
        self._gene_mat = img[self.y_start:, self.x_start:]

    @staticmethod
    def _load_gef(file):
        """
        Sepeedup version that only for gef file format
        """
        chunk_size = 512 * 1024
        with h5py.File(file, "r") as fh:
            dataset = fh["/geneExp/bin1/expression"]

            if not dataset[...].size:
                clog.error("The sequencing data is empty, please confirm the {} file.".format(file))
                raise Exception("The sequencing data is empty, please confirm the {} file.".format(file))

            min_x, max_x = dataset.attrs["minX"][0], dataset.attrs["maxX"][0]
            min_y, max_y = dataset.attrs["minY"][0], dataset.attrs["maxY"][0]
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            img = np.zeros((height, width), np.uint8)
            img.fill(0)

            for step in range(dataset.size // chunk_size + 1):
                data = dataset[step * chunk_size: (step + 1) * chunk_size]
                parse_gef_line(data, img)

        return (
            min_x,
            min_y,
            img,
        )

    @staticmethod
    def gef_gef_shape(file):
        with h5py.File(file, "r") as fh:
            dataset = fh["/geneExp/bin1/expression"]

            if not dataset[...].size:
                clog.error("The sequencing data is empty, please confirm the {} file.".format(file))
                raise Exception("The sequencing data is empty, please confirm the {} file.".format(file))

            min_x, max_x = dataset.attrs["minX"][0], dataset.attrs["maxX"][0]
            min_y, max_y = dataset.attrs["minY"][0], dataset.attrs["maxY"][0]
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            return width, height

    def detect_feature(self, ref: list, chip_size: float):
        """ track lines detection, matrix data: chip area recognition for registration """
        from cellbin2.matrix.box_detect import detect_chip_box
        from cellbin2.matrix.index_points_detect import detect_cross_points

        self._template = detect_cross_points(ref, self._gene_mat)
        self._chip_box = detect_chip_box(self._gene_mat, chip_size)

    def check_standards(self, gs: GeneticStandards):
        # TODO
        #  gs
        pass

    @property
    def template(self, ):
        return self._template

    @property
    def chip_box(self, ):
        return self._chip_box

    @property
    def heatmap(self, ):
        """ gray scale heatmap: for registration """
        return self._gene_mat

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
    
    if os.path.isdir(output_path):
        matrix_name = os.path.basename(input_path).split('.')[0]
        output_path = os.path.join(output_path, f"{matrix_name}.cgef")
        logger.warning(f"Output is a directory, saving CGEF to: {output_path}")
    
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

def is_semantic_mask(mask: np.ndarray) -> bool:
    if mask.dtype != np.uint8:
        return False
    uniq = np.unique(mask)
    return np.array_equal(uniq, [0, 1]) or np.array_equal(uniq, [0, 255])

def main():
    parser = argparse.ArgumentParser(description="Generate cellbin matrix (CGEF) from expression matrix and cell segmentation mask")
    parser.add_argument("-i", "--input", required=True, help="Input file path (.gem.gz or .gef)")
    parser.add_argument("-o", "--output", required=True, help="Output CGEF file path")
    parser.add_argument("-m", "--mask", required=True, help="Cell segmentation mask file path")
    parser.add_argument("-b", "--block-size", type=int, nargs=2, default=[256, 256], help="Block size for CGEF generation (default: 256 256)")
    
    args = parser.parse_args()
    
    logger.info(f"Loading mask: {args.mask}")
    mask = tifffile.imread(args.mask)
    
    if is_semantic_mask(mask):
        logger.info("Detected semantic mask.")
    else:
        logger.info("Detected instance mask. Converting to semantic...")
        mask = instance2semantics(mask)
    
    tmp_mask_path = os.path.join(tempfile.gettempdir(), "converted_mask.tif")
    tifffile.imwrite(tmp_mask_path, mask)

    try:
        ret = generate_cellbin(args.input, args.output, tmp_mask_path, args.block_size)

    finally:
        if os.path.exists(tmp_mask_path):
            os.remove(tmp_mask_path)
            logger.info(f"Temporary mask file deleted: {tmp_mask_path}")
    
    if ret != 0:
        logger.error("Failed to generate cellbin matrix")
    else:
        logger.info("Successfully generated cellbin matrix")
    
    exit(ret)

if __name__ == "__main__":
    main()