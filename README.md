# CS445-Face-Inpainting

This repository contains an implementation of the paper [Globally and Locally Consistent Image Completion](https://arxiv.org/pdf/1704.05838.pdf). The goal of this project is to inpaint masked faces in a natural and seamless manner, resulting in visually appealing final images.

## Dataset

Download the public CelebA dataset by running the following command:

```shell
mkdir data_faces && wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip

Dataset
Download the public CelebA dataset by running the following command:

bash
Copy code
mkdir data_faces && wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
```

## File Structure & description
```
.
├── config                   - config for parsing network 
├── data_faces               - sample original data 
├── model_trained            - trained model
├── preprocessed_images      - sample preprocessed data 
├── pretrained_model
│   ├── parset_00100000.pt   - pretrained model from https://github.com/easternCar/Face-Parsing-Network
├── utils
│   ├── __init__.py
│   ├── evaluate.py            - evaluation functions included
│   ├── models.py              - generator, discriminator model defined
│   └── network_seq_contour.py - parsing network model instance
├── preprocess.py
├── .gitattributes
├── README.md
├── requirement.txt
├── prepare_dataset.ipynb    - responsible for preprocessing the dataset and preparing it for training
├── final_training.ipynb     - contains the main training loop for the face inpainting model
├── evaluate.ipynb           - evaluate the performance of the trained model using various metrics like PSNR and SSIM
├── face_detect.py           - Final script to call face detector API
├── face_detector.ipynb      - Implementation attempts for face detector algorithm, experimentation with face detector API calls
├── combine_result.ipynb     - demonstrates the process of masking faces and inpainting them using the trained model
└── result.csv               - training evaluation result
```
## Setup
To set up the environment and install the required packages, run:

```shell
pip install -r requirements.txt
```

After installing the required packages, follow the instructions in each Jupyter notebook to preprocess the data, train the model, and evaluate its performance.

## Detail implementaion

* Data Preparation:
    * The implementation uses the FaceCompletionDataset class, a public dataset which is called CelebA . We applied a random block mask when loading a batch from the data loader. 
* Methodology
    * Model Details
    * Generator 
        * Gated CNN Generator - which have demonstrated improved performance over traditional CNNs for the task of image inpainting. 
            * Coarse Network
                * Encoder-decoder structure with a bottleneck
                * GatedConv2D layers for better feature selection
                * TransposeGatedConv2D layers for upsampling
            * Refinement Network
                * Similar architecture as the Coarse Network
                * Fine-tunes the output from the Coarse Network
    * Discriminator
        * Consists of several downsampling layers (Conv2D, InstanceNorm2D, and LeakyReLU)
        * Utilizes a custom discriminator block for flexible configuration
        * Employs a stride of 1 in the last downsampling layer and a ZeroPad2D layer followed by a Conv2D layer to output the final prediction
    * Loss Functions:
        * Adversarial Loss: Encourages the generator to produce realistic images that can fool the discriminator.
        * Contextual Loss(Reconstruction Loss): Measures the difference between the ground truth image and the in-painted image.
        * Perceptual Loss(pretrained parsing network): Ensures that the generated inpainting respects the semantic structure of the image.
    * Optimizers and Learning Rate Schedulers:
        * Adam optimizers are used for both the generator and the discriminators with learning rates of 0.0001.
        * Reduce learning rate for every fifth epoch


 

