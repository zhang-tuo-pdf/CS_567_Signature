# Handwriting Verification Using Siamese Neural Networks

## Introduction
In this project, we utilize a Siamese Neural Network (SNN) to build a machine learning-based handwriting verification system. An SNN consists of two identical sub-networks designed to process pairs of input images simultaneously. By feeding the SNN with two signature images, it calculates the contrastive loss between them to assess their similarity. Based on this analysis, the model determines whether the two signatures are genuine or forged.

## Datasets
- **ICDAR**: International Conference on Document Analysis and Recognition dataset 2011.
- Download ICDAR dataset from https://drive.google.com/drive/folders/1hFljH9AKhxxIqH-3fj72mCMA6Xh3Vv0m
- **MSDS**: Manuscript Signature Detection System dataset.
- MSDS dataset is available at https://github.com/HCIILAB/MSDS

## Models
- **CNN**: Custom Convolutional Neural Network designed for feature extraction.
- **ResNet**: Residual Network that leverages deep learning for accurate signature verification.

## Usage

To run the project:

1. Download the desired dataset
2. Configure the `data_loading()` function in `main.py` which locates on the very top with the correct directory paths.
3. In the main execution block (if __name__ == "__main__" section) which locates on the very end of `main.py`, set the `dataset` and `encoder` variables to either 'icdar', 'msds', 'cnn', or 'resnet' as desired.
4. Optionally, adjust the batch size, device, learning rate, and number of training epochs in the script.
5. Execute the script: `main.py`
