# CSRefiner

A lightweight framework for fine-tuning cell segmentation models with small datasets, optimized for spatial transcriptomics (CellBin). Efficiently adapt pre-trained models to new tissue types or staining conditions using minimal annotations. Achieve higher accuracy in cell boundary detection with just a few training samples. Supporting:

- **Cellpose**
- **CellBin v3**

Supports multiple image types (e.g., HE, SS), with options for training from scratch or using pretrained models.

The project pipeline is as follow:

![flow chart](./docs/images/flow_chart.png)

---

## Installation
```bash
git clone https://github.com/STOmics/CSRefiner.git
cd CSRefiner
```
Install dependencies (recommended in a virtual environment):

```bash
pip install -r requirements.txt
```
The Cellpose-SAM related processes require a separate environment. Create another virtual environment and install the dependencies:

```bash
conda env create -f environment_cellpose.yml
```
## Prepare Your Data

### Data Selection
Select several (at least `20` are recommended to ensure the fine-tuning effect) `256×256`-sized small images from the entire chip image to be segmented and annotate them as training sets. The selected small images should be representative, especially including parts where the current segmentation results are not satisfactory (such as the hippocampus of the brain).

If necessary, use `tools/crop_image.py` to crop the chip image into `256×256`-sized small images.

```bash
python tools/crop_image.py \
  -i path/to/raw_images \
  -o path/to/output_patches \
  -s 256
```
The default value of the -s parameter is 256.

Input file format must be `.tif`, and the suffixes must be `"-img.tif"` or `"-mask.tif"`.

If filename contains coordinates like `chipA-x512_y512_w256_h256-img.tif,` cropping will retain correct spatial origin.

Output format: `chip_id-xX_yY_wW_hH-img/mask.tif`, useful for traceability.

### Data Annotation

If your data is not annotated, here is a [tutorial](./docs/Tutorial%20on%20Cell%20Annotation%20Using%20Qupath.md) for manual cell annotation using Qupath.


### Format

Images and masks format should be `.tif`.

Masks format for `Cellpose` can only be `instance mask`.

If necessary, use `tools/mask_convert.py` to convert mask formats:

```bash
# Instance -> Semantic
python tools/mask_convert.py -i input_folder -o output_folder -m i2s

# Semantic -> Instance
python tools/mask_convert.py -i input_folder -o output_folder -m s2i
```

### Naming Requirements

The only difference between the file names of each image and its corresponding mask is the suffix `"-img.tif"` and `"-mask.tif"`.

## Generate a Training List

```bash
python tools/make_trainset_txt.py \
  -img dataset/images \
  -mask dataset/masks \
  -o trainset_list/my_trainset_list.txt
```

This txt file will be used as a necessary parameter input for subsequent fine-tuning.

## Start Fine-Tuning

Download link for the v3 pre-trained models: https://bgipan.genomics.cn/#/link/2AZUv6JJfC4KqL74Rrn0

Password: 8eGM

Pre-trained models included:

- DAPI/ssDNA: cellseg_bcdu_SHDI_221008_tf.hdf5

- HE: cellseg_bcdu_H_240823_tf.hdf5

### Run the main script:

```bash
python run_finetune.py \
  -m model_name/cellpose/or/v3/or/cpsam \
  -t stain/type/ss/or/he \
  -f trainset_list/my_trainset_list.txt \
  -p cyto/or/pretrained/model/path \
  -r 0.9 \
  -b 8 \
  -v 16 \
  -e 100
```

### Required Parameter

| Parameter | Description |
|:----:|:----------:|
|  -m   | Model name: `cellpose` or `v3` or `cpsam`  |
|  -t   | Image type: `ss` or `he` |
|  -f   | Path to `.txt` training list     |

### Optional Parameter

| Parameter | Default | Description |
|:----:|:----------:|:----------:|
|  -p   | None |Pretrained model path/name or scratch. This parameter is not required for fine-tuning cpsam  |
|  -r   | 0.9 |  Train/validation split ratio     |
|  -b   | 6  |Training batch size     |
|  -v   | 16 |Validation batch size    |
|  -e   | 500 | Number of training epochs. For the v3 model, due to its early stopping mechanism, it is the maximum number of training rounds. For the Cellpose model without early stopping, Cellpose officially recommends training for 100 epochs, and it may help to use more epochs, especially when you have more training data.     |

Example: Fine-tuning the Cellpose Model

```bash
python run_finetune.py \
  -m cellpose \
  -t ss \
  -f trainset_list/C05073F4_ss_trainset_list.txt \
  -p cyto \
  -e 100
```
Example: Fine-tuning the Cellpose-SAM Model
```bash
python run_finetune.py \
  -m cpsam \
  -t ss \
  -f trainset_list/C05073F4_ss_trainset_list.txt \
  -e 100
```
Example: Fine-tuning the V3 Model

```bash
python run_finetune.py \
  -m v3 \
  -t ss \
  -f trainset_list/C05073F4_ss_trainset_list.txt \
  -p weights/cellseg_bcdu_SHDI_221008_tf.hdf5 \
  -e 500
```

### Output

After training, outputs are saved to `finetuned_models/model_timestamp/`:

`.pth or .hdf5`: Finetuned model

`train_log.json`: Training/validation loss log

`loss_curve.png`: Training/validation loss plot

## Cell Segmentation Using Fine-Tuned Models

### v3

```bash
python src/segmentor/v3_segmentor.py \
-i input/img/path \
-o /path/to/output/floder \
-p /path/to/model \
-g gpu_index 
```
The input path supports both large-scale images (e.g., whole-slide images) and cropped tiles, as well as batch processing via file or folder paths.

### Cellpose

#### Option 1: Use the official Cellpose CLI

```bash
python -m cellpose \
--dir /path/to/image/floder/or/file \
--pretrained_model /path/to/finetuned/model \
--chan 0 \
--save_tif \
--use_gpu \
--savedir /path/to/output/floder
```
Parameter Description:
| Parameter | Description |
|:----------:|:-------------:|
| --dir  | Path to the  input images folder|
| --pretrained_model | Fine-tuned model path (you can also enter `cyto/cyto3` to use the official pre-trained model)|
| --chan | Segmentation channel, default is 0|
| --save_tif | Segmentation results are saved in `.tif` format |
| --use_gpu | Use GPU |
| ---savedir | Output path |

For more parameters, see: [Cellpose CLI — cellpose 3.1.1.1-24-g3864748 documentation,](https://cellpose.readthedocs.io/en/latest/cli.html) which can be selected according to actual needs.

#### Option 2: Use the script `/src/segmentor/cellpose_segmentor.py`

Cellpose official CLI does not support input of images that are too large. If you want to segment the entire chip image, use the script `/src/segmentor/cellpose_segmentor.py`:

```bash
python src/segmentor/cellpose_segmentor.py \
-i /path/to/image.tif \
-o /path/to/output/floder \
-p /path/to/finetuned/model
```
Similarly, the `-p` parameter also supports inputting `cyto/cyto3` to use the official pre-trained model.

### Cellpose-SAM
#### Option 1: Use the official Cellpose CLI

```bash
python -m cellpose \
--dir /path/to/image/floder \
--pretrained_model /path/to/finetuned/model \
--save_tif \
--use_gpu \
--savedir /path/to/output/floder
```
Parameter Description:
| Parameter | Description |
|:----------:|:-------------:|
| --dir  | Path to the  input images folder|
| --pretrained_model | Fine-tuned model path (Or use the pre-trained cpsam model without this parameter)|
| --save_tif | Segmentation results are saved in `.tif` format |
| --use_gpu | Use GPU |
| ---savedir | Output path |

#### Option 2: Use the script `src/segmentor/cpsam_segmentor.py`

```bash
python src/segmentor/cpsam_segmentor.py \
-i /path/to/image/floder/or/file  \
-o /path/to/output/floder \
-p /path/to/finetuned/model \
-g
```

## Evaluation of cell segmentation results
### Use ImageJ
If there is no groundtruth, you can use ImageJ and select `Image`>`Color`>`Merge Channels` to merge the original image and the model segmentation result, and then observe it with the naked eye.

### Calculate cell segmentation evaluation indexs
Run the script `src/evaluation/cell_eval_multi.py` to calculate the five indicators of `'Precision'`, `'Recall'`, `'F1'`, `'jaccrd'`, and `'dice'` and output the bar chart and box plot at the same time.
```bash
python src/evaluation/cell_eval_multi.py \
-g path/to/groundtruth \
-d path/to/model/results \
-o output/path 
```
Assuming the folder structure is as follows, the `-d` parameter is the path of `dt_path`
```markdown
- dt_path
  - cyto
  - cyto_finetuned
  - v3
```

## Use the segmentation results to obtain the cellbin matrix that combines cell location information
```bash
python src/downstream/generate_cgef.py \
-i input/matrix/path \
-o output/path \
-m path/to/mask \
```
Parameter Description:
| Parameter | Description |
|:----------:|:-------------:|
| -i  | Original gene expression matrix (`.gem.gz`/`.gef` format) path |
| -o | Save path of cellbin matrix combined with cell position information|
| -m | Model segmentation result mask file path|