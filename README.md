# Pigeon Pea Wilt Disease Resistance Analysis

## Overview
This repository contains a series of Jupyter Notebooks and Python scripts developed for the analysis of wilt resistance in pigeon pea plants. The project involves image processing, data augmentation, SMOTE application, and deep learning models for classifying pigeon pea leaf images into various categories based on their health and response to inoculation.

## Repository Structure

- `SMOTE.ipynb`: Jupyter Notebook demonstrating the application of Synthetic Minority Over-sampling Technique (SMOTE) for generating synthetic images to balance the dataset.
- `data_analysis.ipynb`: Notebook containing the analysis of the original dataset, including image counts, formats, and visualization.
- `data_augmentation.ipynb`: This notebook illustrates the process of augmenting the dataset to increase its size and variability.
- `data_preprocessing.py.ipynb`: Python script for preprocessing the data, including cleaning and standardizing images.
- `final_dataset_preparation.ipynb`: Notebook for preparing the final dataset, combining original and augmented images.
- `model_training.ipynb`: Comprehensive notebook detailing the training of deep learning models (ResNet50, VGG16, NASNetMobile) on the processed datasets.

## Project Description

### Objective
To develop a machine learning model capable of accurately classifying pigeon pea leaf images into six categories, helping in the identification and analysis of wilt disease resistance.

### Methodology
- **Data Preprocessing**: Cleaning and standardizing images for uniformity.
- **Data Augmentation**: Enhancing dataset size and diversity.
- **SMOTE**: Balancing the dataset by creating synthetic images.
- **Model Training**: Utilizing ResNet50, VGG16, and NASNetMobile architectures for classification.

### Results
The models were trained and evaluated on two datasets: augmented and SMOTE-enhanced. The performance of each model was assessed based on accuracy and loss metrics, offering insights into the effectiveness of different architectures and dataset preparations.

## Usage

To replicate the analysis or further develop the models:
1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the Jupyter Notebooks in a sequential manner, starting from `data_analysis.ipynb` to `model_training.ipynb`.

## Dependencies
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Authors
Akshansh Agarwal
Bhuvaneswari Kavoori
Nikhil Gupta

## Acknowledgments
This project was inspired by and utilizes data from the research paper "Image Based High throughput Phenotyping for Fusarium
Wilt Resistance in Pigeon Pea (Cajanus cajan)".

