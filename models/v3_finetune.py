import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.utils import Sequence
import tifffile
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

from utils.augmentation import f_rgb2gray, f_percentile_threshold, f_histogram_normalization, f_equalize_adapthist, f_clahe_rgb
import models.models as M
from utils.utils import init_gpu


def f_pre_ssdna(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = f_rgb2gray(img, False)
    img = f_percentile_threshold(img)
    img = f_equalize_adapthist(img, 128)
    img = f_histogram_normalization(img)
    return img

def f_pre_he(img: np.ndarray) -> np.ndarray:
    img = f_clahe_rgb(img)
    img = img.astype(np.float32)
    img = rescale_intensity(img, out_range=(0.0, 1.0))
    return img

def read_txt(file):
    unique_sns = {}
    with open(file, "r") as f:
        data = f.read().splitlines()
    for line in data:
        if line.strip() == "":
            continue
        img_file, label_file = line.split(',')
        sn = os.path.basename(img_file).split("-")[0]
        key = sn if sn[2].isdigit() else "public"
        unique_sns.setdefault(key, []).append([img_file, label_file])
    return unique_sns

def split_train_test(unique_sns, ratio):
    tr_data, tr_mask, val_data, val_mask = [], [], [], []
    for v in unique_sns.values():
        n = int(len(v) * ratio)
        tr_data.extend([x[0] for x in v[:n]])
        tr_mask.extend([x[1] for x in v[:n]])
        val_data.extend([x[0] for x in v[n:]])
        val_mask.extend([x[1] for x in v[n:]])
    return np.array(tr_data), np.array(tr_mask), np.array(val_data), np.array(val_mask)

def plot_loss_curve(log_path, save_path):
    with open(log_path, "r") as f:
        logs = json.load(f)

    epochs = [entry["epoch"] for entry in logs]
    train_loss = [entry["loss"] for entry in logs]
    val_loss = [entry["val_loss"] for entry in logs]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_loss, label='Val Loss', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()


class LossHistory(Callback):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        dct = {
            "epoch": epoch + 1,
            "loss": float(logs.get('loss', 0)),
            "acc": float(logs.get('accuracy', 0)),
            "val_loss": float(logs.get('val_loss', 0)),
            "val_acc": float(logs.get('val_accuracy', 0)),
            "lr": float(tf.keras.backend.get_value(self.model.optimizer.lr))
        }
        self.logs.append(dct)
        with open(self.file, "w") as f:
            json.dump(self.logs, f, indent=2)


class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size, input_shape, pre_fn):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pre_fn = pre_fn
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._load_batch(batch_x, batch_y)

    def on_epoch_end(self):
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def _load_batch(self, batch_x, batch_y):
        xs = np.zeros((self.batch_size,) + self.input_shape, dtype=np.float32)
        ys = np.zeros((self.batch_size,) + self.input_shape[:2], dtype=np.float32)

        for i in range(len(batch_x)):
            img = tifffile.imread(batch_x[i])
            mask = tifffile.imread(batch_y[i])
            mask = np.clip(mask, 0, 1)
            img = self.pre_fn(img)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            xs[i] = img
            ys[i] = mask.astype(np.float32)
        return xs, ys


def train(args):
    input_shape = (256, 256, 3) if args.stain_type == "he" else (256, 256, 1)
    pre_fn = f_pre_he if args.stain_type == "he" else f_pre_ssdna

    init_gpu()

    unique_sns = read_txt(args.txt_file)
    tr_data, tr_mask, val_data, val_mask = split_train_test(unique_sns, args.ratio)
    print(f"Number of training samples: {len(tr_data)}, validation samples: {len(val_data)}")

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    model_save_path = os.path.join("finetuned_models", f"v3_{args.stain_type}_{current_time}")
    os.makedirs(model_save_path, exist_ok=True)

    json_log_path = os.path.join(model_save_path, "train_log.json")
    json_logger = LossHistory(json_log_path)

    model_name = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"

    if args.pretrained_model.lower() == "scratch":
        print("Training from scratch")
        model = M.BCDU_net_D3(input_size=input_shape)
    else:
        print(f"Fine-tuning from {args.pretrained_model}")
        model = tf.keras.models.load_model(args.pretrained_model)

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, model_name),
        save_best_only=False,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,
        mode='min',
        min_delta=1e-4,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        verbose=1,
        min_delta=1e-4,
        mode='min'
    )

    history = model.fit(
        x=DataGenerator(tr_data, tr_mask, args.batch_size, input_shape, pre_fn),
        validation_data=DataGenerator(val_data, val_mask, args.val_batchsize, input_shape, pre_fn),
        epochs=args.nb_epoch,
        callbacks=[checkpoint, early_stop, reduce_lr, json_logger],
        verbose=1
    )

    print("Training finished.")
    plot_loss_curve(json_log_path, model_save_path)