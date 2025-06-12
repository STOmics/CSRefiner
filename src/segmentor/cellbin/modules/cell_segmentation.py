from cellbin.modules import CellBinElement
from cellbin.dnn.cseg.detector import Segmentation
from cellbin.dnn.cseg.cell_trace import get_trace as get_t
from cellbin.dnn.cseg.cell_trace import get_trace_v2 as get_t_v2
from utils import clog
import os
import tf2onnx
import tensorflow as tf


class CellSegmentation(CellBinElement):
    def __init__(self, model_path, gpu="-1", num_threads=0):
        """
        Args:
            model_path(str): network model file path
            gpu(str): gpu index
            num_threads(int): default is 0,When you use the CPU,
            you can use it to control the maximum number of threads
        """
        super(CellSegmentation, self).__init__()

        self._MODE = "onnx"
        self._NET = "bcdu"
        self._WIN_SIZE = (256, 256)
        self._INPUT_SIZE = (256, 256, 1)
        self._OVERLAP = 16

        self._gpu = gpu
        self._model_path = model_path
        self._num_threads = num_threads

        if model_path.endswith(".hdf5"):
            onnx_model_path = model_path.replace(".hdf5", ".onnx")
            if not os.path.exists(onnx_model_path):
                print(f"Converting {model_path} to {onnx_model_path}")
                model = tf.keras.models.load_model(model_path)
                model_proto, _ = tf2onnx.convert.from_keras(model, output_path=onnx_model_path)
                print("Model conversion completed.")
            self._model_path = onnx_model_path

        self._cell_seg = Segmentation(
            net=self._NET,
            mode=self._MODE,
            gpu=self._gpu,
            num_threads=self._num_threads,
            win_size=self._WIN_SIZE,
            intput_size=self._INPUT_SIZE,
            overlap=self._OVERLAP
        )
        clog.info("start loading model weight")
        self._cell_seg.f_init_model(model_path=self._model_path)
        clog.info("end loading model weight")

    def run(self, img):
        """
        run cell predict
        Args:
            img(ndarray): img array

        Returns(ndarray):cell mask

        """
        clog.info("start cell segmentation")
        mask = self._cell_seg.f_predict(img)
        clog.info("end cell segmentation")
        return mask

    @staticmethod
    def get_trace(mask):
        """
        2023/09/20 @fxzhao 对大尺寸图片采用加速版本以降低内存
        """
        if mask.shape[0] > 40000:
            return get_t_v2(mask)
        else:
            return get_t(mask)
