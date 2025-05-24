# Title
 Topic-aware transformer with hierarchical prompting learning for multi-label image classification
# Description
 TATHPL (Topic-Aware Transformer with Hierarchical Prompting Learning) is a novel model designed to enhance multi-label image classification by embedding hierarchical topic information into the feature extraction process. It introduces multiple prompt tokens that are strategically inserted into different transformer layers, allowing the model to capture semantic hierarchies and global contextual dependencies. This approach leverages the self-attention mechanism to propagate topic cues effectively, while maintaining a lightweight design with minimal parameter overhead.

# Dataset Information
MSCOCO:It contains 122,218 images, with 82,081 images in the training set and 40,137 images in the test set used for model evaluation. The dataset includes 80 object categories commonly found in scenes, with each image annotated with an average of 2.9 labels.

NUS-WIDE:NUS-WIDE is a large-scale real-world web-based image dataset containing 269,648 images and 81 visual concepts. We trained our model on 161,789 images and evaluated it on 107,859 images.

Corel5k:Corel5k is a widely used multi-label dataset consisting of 4999 images, with 4500 images designated for the training set and 499 images for the validation set. The dataset is annotated with 260 categories.

# Code Information
The codebase is organized into two main directories:

Topic Model:

This folder contains the script LDA_Corel5k.py, which applies Latent Dirichlet Allocation (LDA) to perform topic clustering on label combinations from all training samples in the Corel5k dataset. The output is a topic distribution for each training instance, serving as semantic guidance for downstream tasks

TATHPL:

This directory includes the implementation of our proposed model and its training pipelines across multiple datasets:

coco_ngpu.py: Trains and evaluates the proposed model on the MS-COCO dataset using multi-GPU support.

Corel5k_main.py: Trains and evaluates the model on the Corel5k dataset.

nus_ngpu.py: Trains and evaluates the model on the NUS-WIDE dataset.

helper_function.py: Provides utilities for dataset preparation, DataLoader creation, and evaluation metric computation.

loss.py: Implements a set of commonly used loss functions, including the Asymmetric Loss adopted in our method.

losses.py: Defines the joint loss functions used in our approach, which include the main multi-label classification loss and an auxiliary topic classification loss.

model_learn.py: Contains the model architecture definition and relevant initialization routines.

# Implementation Steps

1. Dataset Download
Corel5k: All necessary files for this dataset are provided in the GitHub repository.
MSCOCO: Download the COCO 2014 dataset from https://cocodataset.org/#download and place it under the MSCOCO directory.
NUS-WIDE: Download the Flickr folder from Kaggle and place it under the NUS-WIDE directory.


2. Data Preprocessing
For each dataset, we first extract the label combinations of all training samples and save them as a text file. The label combination files for the three datasets used in our paper are as follows:
Corel5k/Corel5k/my_train_label.txt for Corel5k
MSCOCO/targets2014/train/train_labels.txt for MSCOCO
NUS-WIDE/mine/train_image_label.txt for NUS-WIDE


3. Multi-granularity Topic Information Extraction
The extracted label combination files are used as input to the topic model. Set the appropriate number of topics (args.n_topics), specify the input file path (args.train_one_hot_vector_path), and define the output path (args.output_path).
For each training sample, the topic with the highest probability is selected as its assigned topic. The final output is a topic distribution file, where each entry is a key-value pair consisting of the image ID and the index of the assigned topic.
Processed topic distribution files are also provided:
Corel5k/target/train/img_to_index2+3(300).txt
MSCOCO/targets2014/train/img_to_index2+6(300).txt
NUS-WIDE/mine/train/img_to_index2(300).txt


4. Model Training and Evaluation
Using the Corel5k dataset as an example:
Ensure that args.data_corel5k points to the Corel5k directory and args.corel5k_num_class is set to 260. Also, verify that the file path in the Corel5k DataLoader within helper_function.py correctly points to:
file = open("../Corel5k/target/train/img_to_index2+3(300).txt")
Then, run the model using the following command:
python Corel5k_main.py 
# System Requirements
This code has been tested on Windows 11 with the following configuration:

CPU:Xeon Gold 614

GPU:Tesla V100 16G

# Requirements
python==3.9.0
pytorch==2.0.0
timm==0.4.12
randaugment
torchvision==0.15.0