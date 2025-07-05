# Multi-Label Food Classification with Deep Learning Ensembles

This project shows our approach of a multi-label food image classification problem using deep learning and ensemble methods, developed as part of an in-class Kaggle competition.

![Sampled Images](./images/random_images.png)

## Project Context

This project was developed as part of the Advanced Machine Learning course for the MSc in Data Science Programme at the International Hellenic University. The coursework was a hands-on, team-based Kaggle competition titled [Food Recognition (Spring 2025)](https://www.kaggle.com/competitions/food-recognition-spring-2025), designed to simulate a real-world computer vision challenge.

Each team was tasked with developing a machine learning pipeline to solve a multi-label food classification problem and participating teams were evaluated based on their model’s micro F1-score on a hidden private test set.

The competition ran a little under three months and our team placed 2nd overall, among 43 Entrants in 16 Teams, having 571 Submissions total.

## The Dataset

The dataset used in this challenge was adapted from the [AIcrowd Food Recognition Benchmark (2022)](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022) and consists of color food images captured in common, unconstrained environments. Each image may contain one or more food items from a predefined list of 498 possible food categories.

    Training images: 39,962
    Test images: 1,000
    Labels: One-hot encoded vectors of length 498
    Image format: Color JPG images of varying resolutions

## Project Highlights

**Deep Learning Models**: 
Fine-tuned several Transformer-based and Convolutional models on image data using task-specific augmentations and optimization strategies.

**Ensemble Learning**: 
Combined diverse architectures using weighted voting. Our best-performing ensemble included 10 models across 3 architectures.

**Exploratory Data Analysis**: 
Conducted thorough analysis of label distributions, image sizes and normalization metrics as well as post-hoc submission analysis.

**Reproducible & Modular Design**: 
Clear code structure with well-commented cells and markdown explanations for each stage of the pipeline. Easily extensible for new models or datasets.

## Notebooks Included

- **Individual_Notebook.ipynb**: End-to-end pipeline for training, validating and testing a single model. This notebook produced the best individual model.
- **Ensemble_Notebook.ipynb**: Loads multiple trained models, applies ensemble strategies, analyzes model influence and generates final submissions. This notebook produced the best ensemble.
- **EDA_Notebook.ipynb**: Explores image properties, class distributions and test/train differences. Also includes analysis of prediction outputs and error cases post-submission.

## Full Report

For those interested in a deeper dive, the repository includes a full written report outlining our approach, experimentation process, model configurations, ensembling strategies and results. It provides insight into what worked (and what didn’t) throughout the project, including lessons learned during the competition.
