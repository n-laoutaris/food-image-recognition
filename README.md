# Multi-Label Food Classification with Deep Learning Ensembles

This project approaches a multi-label food image classification problem using deep learning and ensemble methods.

![Sampled Images](./images/random_images.png)

## The Problem

Recognizing food items from images is a challenging and far from trivial problem in computer vision. Its importance is particularly apparent in contexts like dietary tracking or nutrition logging. Unlike many typical image classification tasks, food recognition is inherently multi-label; a single dish may contain multiple components, where some can be subtle or occluded.

This project tackles the real-world challenge of building a strong and scalable multi-label classifier that performs well in everyday, messy environments like kitchens and dinner tables, using photographs that are taken in less than ideal circumstances, mimicking how one would photograph their food right before (or even during) their meal.

My ultimate personal goal is to use the final pipeline to classify my own meals using a collection of personal photographs.

## The Dataset

The dataset used was adapted from the [AIcrowd Food Recognition Benchmark (2022)](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022) and consists of color food images captured in common, unconstrained environments. Each image may contain one or more food items from a predefined list of 498 possible food categories.

    Training images: 39,962
    Test images: 1,000
    Labels: One-hot encoded vectors of length 498
    Image format: Color JPG images of varying resolutions

## Methodology

**Exploratory Data Analysis**:
Began with Exploratory data analysis to understand the dataset’s characteristics like image size, label distributions etc.

**Literature Review**:
Researched any recent relevant papers on the topic of image recognition to identify the cutting-edge pre-trained backbones with the best performance, intending to use them through Transfer Learning.

**Experimentation**: 
Fine-tuned several Transformer-based and Convolutional models on the image data at hand, using task-specific augmentations and optimization strategies. Meticulous documentation of results was necessary at this stage, in order to guide iterative improvements.

**Ensembling**: 
After training a diverse set of models, they were combined using weighted voting and other ensembling methods.

**Post-hoc Analysis**: 
Ensemble results were evaluated with hand-crafted metrics such as disagreement rate and decisiveness. Reiterated through the pipeline based on those observations, making the necessary changes.

## Results and Key Takeaways

- The best individual model (ViT) achieved a micro F1-score of **0.583**. The dataset was not large enough, taking the complexity of the task into account, to be able to fine-tune it well enough for a more spectacular score.
- The final ensemble (10 Transformer-based models) improved the score to **0.601**. Ensembling showed a consistent performance boost, but not all models contributed positively. Careful selection and weighting proved crucial.
- Significant performance gaps remained for visually ambiguous items, suggesting the need for contextual cues.
- Careful use of limited resources is a very realistic limitation when tackling real-world problems with Machine Learning.

## Future Work

- Experiment with **self-supervised pretraining** on food-specific datasets to improve generalization on underrepresented classes.
- Introduce **label co-occurrence modeling** or structured prediction techniques to better handle ambiguous multi-label cases.
- Explore **finer-grained segmentation or object detection** as a pipeline extension.

### In progress

- Test the ensemble again using personally shot photographs of my own meals, as the ultimate test of generalization.

## Repository Contents

- `EDA_Notebook.ipynb`: Explores image properties and label distributions. Also includes analysis of prediction outputs and error cases post-submission.
- `Individual_Notebook.ipynb`: End-to-end pipeline for training, validating and testing a single model.
- `Ensemble_Notebook.ipynb`: Loads multiple trained models, applies ensemble strategies, analyzes model influence.
- `Report.pdf`: A detailed written report for those interested in a deeper dive. Explains methodology, experiments, results and lessons learned.
- `requirements.txt`: Lists all Python dependencies required to run the notebooks.

## Project Context

This project was developed as part of the Advanced Machine Learning course for the MSc in Data Science Programme at the International Hellenic University. The coursework was a hands-on, team-based Kaggle competition titled [Food Recognition (Spring 2025)](https://www.kaggle.com/competitions/food-recognition-spring-2025).

Teams were evaluated based on their model’s micro F1-score on a hidden private test set. The competition ran a little under three months and our team placed 2nd overall, among 43 Entrants in 16 Teams, having 571 Submissions total.
