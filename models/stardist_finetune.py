import os
import sys
import json
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from stardist.models import StarDist2D, Config2D
from csbdeep.utils import normalize
from stardist import fill_label_holes


def load_training_data(txt_file, ratio=0.9):
    image_paths, mask_paths = [], []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    image_path, mask_path = parts[0].strip(), parts[1].strip()
                    if os.path.exists(image_path) and os.path.exists(mask_path):
                        image_paths.append(image_path)
                        mask_paths.append(mask_path)
    if ratio < 1.0:
        train_img, val_img, train_mask, val_mask = train_test_split(
            image_paths, mask_paths, test_size=1 - ratio, random_state=42
        )
        return train_img, train_mask, val_img, val_mask
    else:
        return image_paths, mask_paths, [], []


def prepare_data(image_paths, mask_paths, stain_type):
    X, Y = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        color_mode = 'rgb' if stain_type == 'he' else 'grayscale'
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode=color_mode)
        img = tf.keras.preprocessing.image.img_to_array(img)
        if color_mode == 'grayscale':
            img = img.squeeze()
        img = normalize(img, 1, 99.8, axis=(0, 1))  # consistent with StarDist notebook

        mask = tf.keras.preprocessing.image.load_img(mask_path, color_mode='grayscale')
        mask = tf.keras.preprocessing.image.img_to_array(mask).squeeze().astype(np.uint16)
        mask = fill_label_holes(mask)

        X.append(img)
        Y.append(mask)
    return np.array(X), np.array(Y)


def plot_loss_curve(log_path, save_path):
    with open(log_path, "r") as f:
        logs = json.load(f)

    epochs = list(range(1, len(logs['train']) + 1))
    train_loss = logs['train']
    val_loss = logs['val']

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


def train(args):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    model_save_path = os.path.join("finetuned_models", f"stardist_{args.stain_type}_{current_time}")
    os.makedirs(model_save_path, exist_ok=True)
    json_log_path = os.path.join(model_save_path, "train_log.json")
    best_weights_path = os.path.join(model_save_path, "best_weights.h5")

    train_img, train_mask, val_img, val_mask = load_training_data(args.txt_file, args.ratio)
    if not train_img:
        logging.error("No training data found!")
        return

    logging.info(f"Training samples: {len(train_img)}")
    logging.info(f"Validation samples: {len(val_img)}")

    # Load model: from scratch, from pretrained name, or existing trained model path
    if args.pretrained_model == 'scratch':
        config = Config2D(n_channel_in=3 if args.stain_type == 'he' else 1)
        model = StarDist2D(config, name=f'stardist_{args.stain_type}', basedir=model_save_path)
        logging.info("Initialized StarDist model from scratch.")
    else:
        if args.pretrained_model.endswith('.h5'):
            config = Config2D(n_channel_in=3 if args.stain_type == 'he' else 1)
            model = StarDist2D(config, name=f'stardist_{args.stain_type}', basedir=model_save_path)
            model.keras_model.load_weights(args.pretrained_model)
            logging.info(f"Loaded weights from .h5 file: {args.pretrained_model}")
        else:
            model = StarDist2D.from_pretrained(args.pretrained_model)
            logging.info(f"Loaded StarDist pretrained model: {args.pretrained_model}")

    # Prepare data
    X_train, Y_train = prepare_data(train_img, train_mask, args.stain_type)
    X_val, Y_val = prepare_data(val_img, val_mask, args.stain_type) if val_img else (None, None)

    best_val_loss = float('inf')
    best_epoch = 0
    history_log = {"train": [], "val": []}
    patience = 30
    epochs_no_improve = 0

    steps_per_epoch = max(10, len(X_train) // args.batch_size)

    try:
        for epoch in range(1, args.nb_epoch + 1):
            logging.info(f"Epoch {epoch}/{args.nb_epoch}")

            history = model.train(
                X_train, Y_train,
                validation_data=(X_val, Y_val) if X_val is not None else None,
                epochs=1,
                steps_per_epoch=steps_per_epoch,
            )

            train_loss = history.history['loss'][0]
            val_loss = history.history.get('val_loss', [float('inf')])[0]

            history_log["train"].append(train_loss)
            history_log["val"].append(val_loss)

            with open(json_log_path, "w") as f:
                json.dump(history_log, f, indent=2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                model.keras_model.save_weights(best_weights_path)
                logging.info(f"New best weights saved at epoch {epoch} (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break

        #Load best weights
        model.keras_model.load_weights(best_weights_path)
        logging.info(f"Loaded best weights from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

        # Save model
        if args.pretrained_model.lower() == 'scratch':
            model_source = 'scratch'
        else:
            model_source = os.path.splitext(os.path.basename(args.pretrained_model))[0]

        weights_filename = f"{model_source}_best_epoch{best_epoch}.h5"
        weights_path = os.path.join(model_save_path, weights_filename)
        model.keras_model.save_weights(weights_path)
        logging.info(f"Final best weights saved to: {weights_path}")

        if os.path.exists(best_weights_path):
            os.remove(best_weights_path)
        logging.info(f"Deleted temporary file: {best_weights_path}")

        plot_loss_curve(json_log_path, model_save_path)
        logging.info("StarDist training completed.")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise