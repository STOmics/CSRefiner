import argparse
import importlib
import importlib.util
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model_module(model_name: str):
    module_path = Path(__file__).parent / "models" / f"{model_name}_finetune.py"
    
    if not module_path.exists():
        available = [f.stem.replace('_finetune', '') 
                     for f in (module_path.parent).glob('*_finetune.py')]
        raise FileNotFoundError(
            f"Model file '{module_path}' not found.\nAvailable models: {available}"
        )

    spec = importlib.util.spec_from_file_location(f"models.{model_name}_finetune", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning script based on the -m parameter.")
    parser.add_argument('-m', '--model', required=True, choices=['v3', 'cellpose', 'cpsam'], help="Model name to finetune")
    parser.add_argument('-t', '--stain_type', choices=['ss', 'he'], required=True, help="Image type: ss or he")
    parser.add_argument('-f', '--txt_file', required=True, help="Path to training list (.txt)")
    parser.add_argument('-p', '--pretrained_model', help="Path to pretrained model (.hdf5), or 'scratch'")
    parser.add_argument('-r', '--ratio', type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument('-b', '--batch_size', type=int, default=6, help="Training batch size")
    parser.add_argument('-v', '--val_batchsize', type=int, default=16, help="Validation batch size")
    parser.add_argument('-e', '--nb_epoch', type=int, default=500, help="Number of training epochs")
    
    args = parser.parse_args()

    logging.info(f"Loading model module for: {args.model}")
    try:
        module = load_model_module(args.model)
        if hasattr(module, "train"):
            logging.info("Starting training...")
            module.train(args)
        else:
            logging.error(f"No 'train' function found in module for model '{args.model}'.")
    except Exception as e:
        logging.error(f"Failed to load or execute module: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()