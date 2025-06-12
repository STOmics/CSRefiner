import argparse
import importlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():

    parser = argparse.ArgumentParser(description="Run fine-tuning script based on the -m parameter.")
    parser.add_argument('-m', '--model', required=True, choices=['v3', 'cellpose'], help="Model name to finetune (e.g., v3, cellpose)")
    parser.add_argument('-t', '--stain_type', choices=['ss', 'he'], required=True, help="Image type: ss or he")
    parser.add_argument('-f', '--txt_file', required=True, help="Path to training list (.txt)")
    parser.add_argument('-p', '--pretrained_model', required=True, help="Path to pretrained model (.hdf5), or 'scratch'")
    parser.add_argument('-r', '--ratio', type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument('-b', '--batch_size', type=int, default=6, help="Training batch size")
    parser.add_argument('-v', '--val_batchsize', type=int, default=16, help="Validation batch size")
    parser.add_argument('-e', '--nb_epoch', type=int, default=500, help="Number of training epochs")
    
    args = parser.parse_args()
    
    try:
        module_name = f"models.{args.model}_finetune"
        logging.info(f"Loading module: {module_name}")
        
        model_module = importlib.import_module(module_name)
        
        if hasattr(model_module, "train"):
            logging.info(f"Starting training with module: {module_name}")
            model_module.train(args)
        else:
            logging.error(f"Module '{module_name}.py' does not have a 'train' function.")
    
    except ModuleNotFoundError:
        logging.error(f"Finetune script '{module_name}.py' not found in the 'models' folder.")

if __name__ == "__main__":
    main()