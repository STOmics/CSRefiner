import os
import json
from datetime import datetime
import numpy as np
from cellpose import io, models
from cellpose.train import train_seg
import matplotlib.pyplot as plt

io.logger_setup()

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

    epochs = list(range(1, len(logs['train']) + 1))
    train_loss = logs['train']
    val_loss = logs['test']

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

def load_cellpose_data(data_files, mask_files):
    images = []
    labels = []
    for img_file, mask_file in zip(data_files, mask_files):
        img = io.imread(img_file)
        mask = io.imread(mask_file)
        images.append(img)
        labels.append(mask)
    return images, labels

def train(args):
    input_shape = (256, 256, 3) if args.stain_type == "he" else (256, 256, 1)

    unique_sns = read_txt(args.txt_file)
    tr_data, tr_mask, val_data, val_mask = split_train_test(unique_sns, args.ratio)
    print(f"Number of training samples: {len(tr_data)}, validation samples: {len(val_data)}")

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    model_save_path = os.path.join("finetuned_models", f"cellpose_{args.stain_type}_{current_time}")
    os.makedirs(model_save_path, exist_ok=True)

    # Load data
    train_images, train_labels = load_cellpose_data(tr_data, tr_mask)
    test_images, test_labels = load_cellpose_data(val_data, val_mask)

    # Initialize Cellpose model
    model = models.CellposeModel(gpu=True)

    n_epochs = args.nb_epoch
    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 1
    model_name = f"finetuned_cpsam_{args.nb_epoch}_epoch"

    json_log_path = os.path.join(model_save_path, "train_log.json")

    model_path, train_losses, test_losses = train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=max(2, len(train_images)), # can change this
        model_name=model_name,
        save_path=model_save_path
    )

    with open(json_log_path, "w") as f:
        json.dump({'train': train_losses.tolist(), 'test': test_losses.tolist()}, f, indent=2)

    print("Training finished.")
    plot_loss_curve(json_log_path, model_save_path)
