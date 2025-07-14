import tensorflow as tf
import os
import pathlib
import tifffile
import numpy as np
import cv2
from typing import Any, Dict, Tuple, Union, List


def init_gpu():
    """Initialize and configure GPU settings."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to prevent TensorFlow from consuming all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"Memory config failed: {e}")

def cell_filter(mask, area_thr: int):
    mask_ = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    for c in contours:
        area = cv2.contourArea(c)
        if area >= area_thr: cv2.fillPoly(mask_, c)
    return mask_


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


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_


def cell_dataset(file_path: str, exts: list):
    img_lst = search_files(file_path, exts)
    return [i for i in img_lst if 'img' in i ]


# def auto_make_dir(file_path: str, src: str, output: str):
#     i_output_path = os.path.dirname(file_path.replace(src, output))
#     if not os.path.exists(i_output_path): os.makedirs(i_output_path)
    
#     return os.path.join(i_output_path, os.path.basename(file_path))
def auto_make_dir(file_path: str, src: str, output: str):
    if os.path.isdir(src):
        rel_path = os.path.relpath(os.path.dirname(file_path), src)
        i_output_path = os.path.join(output, rel_path)
    else:
        i_output_path = output

    os.makedirs(i_output_path, exist_ok=True)
    return os.path.join(i_output_path, os.path.basename(file_path))


def cellpose_channel_detect(mat):
    if mat.ndims == 2:
        return [0, 0]
    else:
        h, w, c = mat.shape
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)


class CBImage(object):
    """ channel image: 单通道图 """

    _SUFFIX_LIST = ['.tif', '.tiff', '.TIF', '.TIFF']
    _SECOND_SUFFIX_LIST = ['.png', '.jpg', '.jpeg']

    def __init__(self, file: Union[str, np.ndarray] = None) -> None:
        """ Init image.

        Args:
            file:
        """
        self._file_path: str = ''
        self._width = 0
        self._height = 0
        self._depth = 16
        self._channel = 1
        self._ndim = 2

        self._image = None

        self._tif_method = True

        if file is not None:
            self.read(file)

    def read(self, file: Union[str, np.ndarray],
             key: int = 0) -> None:
        """
        Args:
            file:
            key:

        Returns:

        """
        if isinstance(file, (str, pathlib.PosixPath, pathlib.WindowsPath)):
            assert os.path.isfile(file), "File not exists."
            suffix = os.path.splitext(file)[1]
            self._file_path = file
            try:

                if suffix in self._SUFFIX_LIST:
                    self._image = tifffile.imread(file, key=key)
                elif suffix.lower() in self._SECOND_SUFFIX_LIST:
                    self._image = cv.imread(file, -1)
                    self._tif_method = False
                else:
                    raise ValueError("File suffix not supported.")

            except Exception as e:
                print("File read error with " + str(e))

        elif type(file) is np.ndarray:
            self._image = file

        self._parse_image()

    def crop_image(self, border: Union[np.ndarray, List, Tuple]) -> 'CBImage':
        """

        Args:
            border: y0, y1, x0, x1

        Returns:

        """
        y0, y1, x0, x1 = border
        if self._ndim == 3:
            new_image = self._image[y0: y1, x0: x1, :]
        else:
            new_image = self._image[y0: y1, x0: x1]

        return CBImage(new_image)

    def resize_image(self, size: Union[int, float, List, Tuple]) -> 'CBImage':
        """

        Args:
            size: int, float, List: [y, x], Tuple: [y, x]

        Returns:

        """
        if isinstance(size, (float, int)):
            new_image = cv.resize(self._image, [int(self._width * size), int(self._height * size)])

        elif isinstance(size, (list, tuple)):
            new_image = cv.resize(self._image, [size[1], size[0]])

        else:
            raise ValueError("Size format error.")

        return CBImage(new_image)

    def trans_image(
            self,
            scale: Union[float, List, Tuple] = None,
            rotate: float = None,
            rot90: int = None,
            offset: Union[Tuple, List] = None,
            dst_size: Union[Tuple, List] = None,
            flip_lr: bool = False,
            flip_ud: bool = False,
            trans_mat: np.matrix = None
    ) -> 'CBImage':
        """ 调用pyvips进行图像操作

        Args:
            scale:
                - list | tuple, [scale_x, scale_y]
            rotate:
            rot90:
            offset: [x, y]
            dst_size: (height, width)
            flip_lr:
            flip_ud:
            trans_mat:

        Returns:

        """
        from .transform import ImageTransform

        it = ImageTransform()
        it.set_image(self._image)

        if trans_mat is not None:
            # TODO
            return

        if flip_lr:
            it.flip(flip_type = 'hor',
                    ret_dst = False
                    )

        if flip_ud:
            it.flip(flip_type = 'ver',
                    ret_dst = False
                    )

        if isinstance(scale, float) or scale is None:
            scale_x = scale_y = scale
        else:
            scale_x, scale_y = scale

        it.rot_scale(
            x_scale = scale_x,
            y_scale = scale_y,
            angle = rotate
        )

        if rot90 is not None:
            it.rot90(rot90, ret_dst = False)

        if offset is not None:
            offset_x, offset_y = offset
            it.offset(
                x_offset = offset_x,
                y_offset = offset_y,
                dst_size = dst_size
            )

        return CBImage(it.to_image())

    def get_channel(self, channel: int = 0) -> 'CBImage':
        """

        Args:
            channel:

        Returns:

        """
        if self._channel == 1:
            return CBImage(self._image)
        else:
            if channel < 0 or channel > self._channel - 1:
                raise ValueError("Channel error.")
            return CBImage(self._image[:, :, channel])

    def _parse_image(self) -> None:
        """

        Returns:

        """
        self._image = np.squeeze(self._image)
        self._depth = 8 if self._image.dtype == "uint8" else 16

        self._ndim = self._image.ndim

        if self._ndim == 3:
            shape = self._image.shape
            if shape[0] in [1, 2, 3, 4]:
                self._image = self._image.transpose(1, 2, 0)

            if not self._tif_method:
                self._image = self._image[:, :, ::-1]

            self._height, self._width, self._channel = self._image.shape
        else:
            self._height, self._width = self._image.shape
            self._channel = 1

    def write(self, file_path: str, compression: bool = True) -> None:
        """

        Args:
            file_path:
            compression:

        Returns:

        """
        self.write_s(self._image, file_path, compression=compression)

    @staticmethod
    def write_s(image, output_path: str, compression=False):
        try:
            if compression:
                if image.nbytes > 4294967295:
                    big_tiff, tile = True, [512, 512]
                else:
                    big_tiff, tile = None, None
                tifffile.imwrite(output_path, image, compression="zlib",
                                 compressionargs={"level": 8}, bigtiff=big_tiff,
                                 tile=tile)
            else:
                tifffile.imwrite(output_path, image)
        except Exception as e:
            print(str(e) + "\n" +
                  "Errors on writing image with compression, try without compression.")
            tifffile.imwrite(output_path, image)

    def to_gray(self) -> 'CBImage':
        """

        Returns:

        """
        if self._ndim == 3:
            image = cv.cvtColor(self._image, cv.COLOR_RGB2GRAY)
            return CBImage(image)

        return self

    @property
    def shape(self, ):
        return self._height, self._width

    @property
    def channel(self, ):
        return self._channel

    @property
    def height(self, ):
        return self._height

    @property
    def width(self, ):
        return self._width

    @property
    def image(self, ) -> np.ndarray:
        return self._image

    @property
    def file_path(self, ) -> str:
        return self._file_path

    @property
    def ndim(self, ):
        return self._ndim

    @property
    def depth(self, ):
        return self._depth

    @property
    def print_info(self, ) -> Dict:
        """

        Returns:

        """
        _dict = dict()
        for k, v in self.__dict__.items():
            if "image" not in k:
                _dict[k[1:]] = v
        return _dict


def cbimread(
    files: Union[str, np.ndarray, os.PathLike, List, Tuple],
    only_np: bool = False,
    **kwargs
) -> Union[np.ndarray, CBImage, List]:
    """ Read files as NumPy or CBImage.

    Args:
        files:
        only_np:
        **kwargs:

    Returns:

    """
    if isinstance(files, (list, tuple)):
        # TODO
        pass
    elif isinstance(files, (str, os.PathLike, np.ndarray)):
        cbi = CBImage(files)
        if only_np:
            return cbi.image
        else: return cbi
    else:
        raise ValueError("File not supported.")


def cbimwrite(
        output_path: str,
        files: Union[np.ndarray, CBImage],
        compression: bool = True
) -> None:
    """

    Args:
        output_path:
        files:
        compression:

    Returns:

    """
    # if not os.path.isfile(output_path):
    #     raise ValueError("File format error.")

    if isinstance(files, np.ndarray):
        CBImage.write_s(files, output_path, compression=compression)
    elif isinstance(files, CBImage):
        files.write(output_path)
    else:
        raise ValueError("File not supported.")
    