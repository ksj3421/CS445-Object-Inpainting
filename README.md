# CS445-Face-Inpainting

This repository contains a implementation this paper (https://arxiv.org/pdf/1704.05838.pdf). The goal of this project is to inpaint masked face in a natural and seamless way, so that the final image looks visually appealing.


* data preprocessing - prepare_dataset.ipynb
how i downloaded dataset
mkdir data_faces && wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip 

origin_data - data_faces/img_align_celeba
preprocessed_data - preprocessed_images
masked_image - masked_images
masks - masks

* training - training.ipynb
