# Multi-Label Food Classification with Deep Learning Ensembles

This repository contains a robust and well-documented solution to a multi-label food image classification task, developed as a group project for the Advanced Machine Learning course during the Spring 2025 semester. The project was built around a real-world-style Kaggle competition, where teams were challenged to recognize multiple food categories from complex images using deep learning.

![Sampled Images](./images/random_images.png)

## Project Highlights

**Deep Learning Models**
Fine-tuned several Swin Transformer, Swin V2, and Vision Transformer (ViT) models on food image data using task-specific augmentations and normalization strategies.

**Ensemble Learning**
Combined diverse models using weighted voting to boost performance. Our best-performing ensemble included 10 models across 3 architectures, achieving the highest public score in the competition.

**Exploratory Data Analysis (EDA)**
Conducted thorough analysis of label distributions, image sizes, and label co-occurrence to inform preprocessing and thresholding strategies.

**Reproducible & Modular Design**
Clear code structure with grouped imports, well-commented cells, and markdown explanations for each stage of the pipeline. Easily extensible for new models or datasets.

## Notebooks Included

- **Individual_Notebook.ipynb**: End-to-end pipeline for training, validating, and testing a single model.
- **Ensemble_Notebook.ipynb**: Loads multiple trained models, applies ensemble strategies, analyzes model influence, and generates final submissions.
- **EDA_Notebook.ipynb**: Explores image properties, class distributions, and test/train differences. Also includes analysis of prediction outputs and error cases post-submission.

## Results

Leaderboard Scores:

    Best ensemble: 0.59534

    Final rank: 2nd place 

