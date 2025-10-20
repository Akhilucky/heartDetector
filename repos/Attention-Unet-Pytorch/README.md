# Attention Unet Pytorch

This repository contains an unofficial PyTorch implementation of the paper `Attention U-Net: Learning Where to Look for the Pancreas` by Zhu et al. The paper introduces an attention-based variation of the U-Net architecture for pancreas segmentation in medical imaging.

Paper: [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)

## Overview

The Attention U-Net model is a modification of the U-Net architecture, incorporating self-attention mechanisms to selectively attend to informative regions in the image during both encoding and decoding. This approach enhances the model's ability to capture fine-grained details and spatial dependencies, leading to improved segmentation performance.

## Dependencies

- Python 3.x
- PyTorch

to install all the dependencies run the following command:

```bash
pip3 install -r requirements.txt
```

## Additional Information

This code is the implemented for the term project of CS 484 Introduction to Computer Vision at Bilkent University. The project is done by [Yiğit Ekin](https://github.com/YigitEkin) and [Arda Eren](https://github.com/arda-eren). The inference of this code on a dataset can be visualized by this [link](https://www.kaggle.com/code/ygtekn/semantic-segmentation-with-attention-unet-pytorch/notebook).
