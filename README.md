# MS-EMD
### Environment
TensorFlow=2.3
Python=3.8
pygame
opencv-python
tqdm
### Directory Hierarchy
```
MSEMD
    \_ datasets: images, training data and test data.
        \_ chars_1500.txt: the most commonly used 1500 characters for experiment.
    \_ fonts: font files.
    \_ results: experiment results.
        \_ concat_for_compare.py: concatenate results for compare.
        \_ fid.py: calculate fid score.
        \_ quantify.py: code for quantification.
    \_ runs: experiment summary for tensorboard.
    \_ src: source code.
        \_ logger.py: log writer.
        \_ networks.py: model structure.
        \_ test.py: code for test.
        \_ train.py: code for training.
        \_ utils.py: utilities.
    font2img.py: generate images use font files.
    get_data_list.py: generate training and test data.
```

### Dataset
**step1.** Run `font2img.py` to generate font images, font_list and char_list.
Configs:
```
img_path: path to save font images
font_path: path to load font files
resol: resolution of images, 80 in default.
```
**step2.** Run `get_data_list.py` to generate dataset which contains training set and test set.
Configs:
```
font_list: fonts to partition training area and test area.
char_list: characters to partition training area and test area.
train_iter: size of training data.
test_iter: size of test data.
content_sample_num: the number of images for content encoder inputs.
style_sample_num: the number of images for style encoder inputs.
image_path: path to load images.
train_save_path: path to save training data.
test_save_path: path to save test data.
style_num: the number of style encoders.
get_data_list(): generate training data for original EMD.
get_multi_style_data_list(): generate training data for MSEMD.
```
> (Optional) Run `font2img.py` and `get_data_list.py` again to generate images and dataset of resolution 256.

### Training
**stage1.** Run `train.py` to train MSEMD model:
```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py
                        --input_dir datasets/XXX/train
                        --output_dir results/XXX/train
                        --save_summary 1
                        --style_num 3
                        --with_dis 1
```
**(Optional) stage2.** Run `train.py` to train local enhancer:
```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py
                        --input_dir datasets/XXX/train_256
                        --output_dir results/XXX/train_256
                        --save_summary 1
                        --epochs 50
                        --input_size 256
                        --batch_size 16
                        --save_size 2 8
                        --style_num 3
                        --with_local_enhancer 1
                        --global_checkpoint results/XXX/train
```
### Test
Run `test.py` to generate images of 4 areas and images of style fusion:
```
CUDA_VISIBLE_DEVICES=0 python src/test.py
                        --input_dir datasets/XXX/test
                        --output_dir results/XXX/test
                        --checkpoint results/XXX/train
                        --style_num 3
CUDA_VISIBLE_DEVICES=0 python src/test.py
                        --input_dir datasets/XXX/test
                        --output_dir results/XXX/interp
                        --checkpoint results/XXX/train
                        --batch_size 5
                        --save_size 5 1
                        --style_num 3
                        --interp 1
```
> Corresponding commands to generate images of resolution 256:
```
CUDA_VISIBLE_DEVICES=0 python src/test.py
                        --input_dir datasets/XXX/test_256
                        --output_dir results/XXX/test_256
                        --checkpoint results/XXX/train_256
                        --input_size 256
                        --style_num 3
                        --with_local_enhancer 1
CUDA_VISIBLE_DEVICES=0 python src/test.py
                        --input_dir datasets/XXX/test_256
                        --output_dir results/XXX/interp_256
                        --checkpoint results/XXX/train_256
                        --input_size 256
                        --batch_size 5
                        --save_size 5 1
                        --style_num 3
                        --with_local_enhancer 1
                        --interp 1
```