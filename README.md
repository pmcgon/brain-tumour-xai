# brain-tumour-xai

## Overview
This project focuses on preprocessing brain MRI images and developing a deep learning model for brain tumor classification. It is split into two main parts: local preprocessing and cloud-based model training/analysis.

## Repository Link
GitHub Repository: [brain-tumour-xai](https://github.com/pmcgon/brain-tumour-xai)

## Project Structure

- `preproc.py`: Script for preprocessing the MRI data including extracting 2D Slices from the 3D .nii files, normalising, cropping, undersampling and splitting data.
- `BrainTumourXAI.ipynb`: Jupyter notebook for training the model and performing XAI analysis on Google Colab.
- `XAI Images` : Folder that contains sample images used in XAI sections of notebook. 


# Table of Contents

1. Dataset Intsallation
2. Local Environment Setup (Preprocessing)
3. Cloud Environment Setup (Google Colab)

# Dataset Installation
1. Downlaod the BraTS dataset from here: https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/

   Scroll to the Data Access and downlaod the "Challenge Data both tasks" dataset (142GB). You will need to install the IBM-ASPERA-Connect which can be downloaded below the downlaod link for the dataset.
2. Extract the BraTS2021_TrainingSet to another folder. Ensure the dataset is organised as so:
   ````bash
   BraTS2021_TrainingSet/
   ├── ACRIN-FMISO-Brain/
   ├── CPTAC-GBM/
   ├── IvyGAP/
   ├── new-not-previously-in-TCIA/
   ├── TCGA-GBM/
   ├── TCGA-LGG/
   ├── UCSF-PDGM/
   └── UPENN-GBM/

# Local Environment Setup (Preprocessing)

## Prequisites

- Anaconda
- Python 3.11

## Usage

1. **Setup Conda Environment**

   First, create a conda environment and install the necessary packages:

   ```bash
   conda create -n mri-preprocessing python=3.11
   conda activate mri-preprocessing
   ```
  
2. **Install Packages**

    Next install the required packages using conda and pip:
   ```bash
   conda install numpy pillow tqdm scikit-learn
   pip install nibabel

3. **Data Preprocessing**

   Place your dataset in the appropriate directory as defined in the script (preprocessing.py). The default directory is C:\BraTS2021_TrainingSet, but it can be changed along with the output directories by changing the variables `base_dataset_path `, `output_path` and `balanced_output_path`.

   The script extracts MRI images from 3D to 2D, crops and normalises the slices, and creates a balanced dataset by undersampling the majority class.
    
    Run the preprocessing script:
    ```bash
    python preproc.py
    ```
    The script will:
    
    - Create a directory structure for the processed data.
    - Extract slices from the MRI scans.
    - Balance the dataset by undersampling the majority class.

    The processed data will be saved in the directory specified by `balanced_output_path`

5. **Zip and Upload Processed Data**

   After preprocessing, zip the processed data and upload it to Google Drive

# Cloud Environment

## Prequisites

1. Upload the preprocessed dataset to Google Drive (as per above).
2. Also upload the `XAI images` to Google Drive. It contains some sample images to use.
3. Open The BrainTumourXAI.ipynb file in Google Colab.
4. Use a T4 GPU runtime instance for the Colab session. This will ensure there is enough resources.
5. Mount the Google Drive
   ```bash
   from google.colab import drive
   drive.mount('/content/drive')
   ```
## Usage
Run the code blocks sequentially

**Note:** If you're running the older model `OriginalModel`, ensure that you adjust the training loop accordingly.



    
