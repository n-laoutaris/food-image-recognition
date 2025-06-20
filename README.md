# Multi-Label Food Classification with Deep Learning Ensembles



![Sampled Images](./random_images.png)

## Project Context

This project was developed as part of the Advanced Machine Learning course at International Hellenic University. The course included a hands-on, team-based Kaggle competition titled [Food Recognition (Spring 2025)](https://www.kaggle.com/competitions/food-recognition-spring-2025), designed to simulate a real-world computer vision challenge.

Each team was tasked with developing a machine learning pipeline to solve a multi-label food classification problem. Each image in this challenge could contain multiple food categories, requiring careful handling of multi-label outputs, thresholds, and prediction logic.

The competition ran over three months, and participating teams were evaluated based on their model’s micro F1-score on a hidden private test set.

Our team placed 2nd overall.

## The Dataset

The dataset used in this challenge was adapted from the [AIcrowd Food Recognition Benchmark (2022)](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022) and consists of color food images captured in common, unconstrained environments. Each image may contain zero or more food items from a predefined list of 498 possible food categories.

    Training images: 39,962
    Test images: 1,000
    Labels: One-hot encoded vectors of length 498
    Image format: Color JPG images of varying resolutions

## Project Highlights

**Deep Learning Models**: 
Fine-tuned several Swin Transformer, Swin V2, and Vision Transformer (ViT) models on food image data using task-specific augmentations and normalization strategies.

**Ensemble Learning**: 
Combined diverse models using weighted voting to boost performance. Our best-performing ensemble included 10 models across 3 architectures, achieving the highest public score in the competition.

**Exploratory Data Analysis (EDA)**: 
Conducted thorough analysis of label distributions, image sizes, and label co-occurrence to inform preprocessing and thresholding strategies.

**Reproducible & Modular Design**: 
Clear code structure with grouped imports, well-commented cells, and markdown explanations for each stage of the pipeline. Easily extensible for new models or datasets.

## Notebooks Included

- **Individual_Notebook.ipynb**: End-to-end pipeline for training, validating, and testing a single model. This notebook produced the best individual model.
- **Ensemble_Notebook.ipynb**: Loads multiple trained models, applies ensemble strategies, analyzes model influence, and generates final submissions. This notebook produced the best ensemble.
- **EDA_Notebook.ipynb**: Explores image properties, class distributions, and test/train differences. Also includes analysis of prediction outputs and error cases post-submission.

## Full Report

For those interested in a deeper dive, the repository includes a full written report outlining our approach, experimentation process, model configurations, ensembling strategies, and results. It provides insight into what worked (and what didn’t) throughout the project — including lessons learned during the competition.

## Results

Our best-performing solution was an ensemble of ten models, combining Swin and ViT models. This blend of architectures helped us capture different “perspectives” on the data and led to a submission F1 score of 0.59204, the highest score across all our runs.

Key takeaways:
- A diverse mix of models beat any single-model strategy.
- Allowing occasional “no label” predictions improved overall performance.
- Small architectural and training tweaks (e.g., input size, augmentations) made a big difference at scale.
