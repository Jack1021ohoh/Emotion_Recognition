# Emotion Recognition from Twitter Text

A comprehensive deep learning project for multi-class emotion classification from Twitter messages using state-of-the-art NLP models.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)

## Overview

This project was developed as the final project for the Data Mining and Knowledge Discovery course. It implements multiple deep learning approaches to classify emotions in Twitter messages across six categories: **anger**, **fear**, **joy**, **love**, **sadness**, and **surprise**.

The project explores various state-of-the-art NLP architectures including LSTM-based models, transformer-based models (BERT, RoBERTa, T5), and traditional machine learning approaches with sentence embeddings.

## Dataset

**Source**: [Emotion Dataset on Kaggle](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)

**Dataset Statistics**:
- Training set: 16,000 samples
- Validation set: 2,000 samples
- Test set: 2,000 samples

**Emotion Distribution** (Training set):
- Joy: 33.5% (5,362 samples)
- Sadness: 29.2% (4,666 samples)
- Anger: 13.5% (2,159 samples)
- Fear: 12.1% (1,937 samples)
- Love: 8.2% (1,304 samples)
- Surprise: 3.6% (572 samples)

The dataset exhibits class imbalance, with joy and sadness being the dominant emotions.

## Methodology

### 1. Exploratory Data Analysis ([eda-of-the-emotion-dataset.ipynb](eda-of-the-emotion-dataset.ipynb))
- Statistical analysis of text length and word count distributions
- N-gram analysis (unigrams, bigrams, trigrams) for each emotion category
- Word cloud generation to visualize common words per emotion
- Class distribution analysis

### 2. Model Development
The project implements and compares multiple approaches:

#### Traditional ML with Embeddings
- Sentence embeddings using `paraphrase-distilroberta-base-v2`
- LightGBM classifier on sentence embeddings
- **Accuracy**: 68%

#### Deep Learning Models
- **LSTM with TensorFlow** ([emotion-recognition-ltsm-tensorflow.ipynb](emotion-recognition-ltsm-tensorflow.ipynb))
- **BERT** ([emotion-recognition-bert.ipynb](emotion-recognition-bert.ipynb))
  - Fine-tuned RoBERTa-base model
  - **Accuracy**: 92.5%
  - **F1-Score**: 0.926 (weighted)
- **T5** ([emotion-recognition-t5.ipynb](emotion-recognition-t5.ipynb))

## Models Implemented

### 1. LightGBM with Sentence Embeddings
- Uses pre-trained sentence transformers for feature extraction
- Fast training and inference
- Baseline performance: 68% accuracy

### 2. LSTM (TensorFlow)
- Recurrent neural network approach
- Sequential text processing
- Implementation details in [emotion-recognition-ltsm-tensorflow.ipynb](emotion-recognition-ltsm-tensorflow.ipynb)

### 3. RoBERTa (Best Performing Model)
- Fine-tuned from `roberta-base` checkpoint
- Training configuration:
  - Epochs: 6
  - Batch size: 32
  - Learning rate: 1e-5
  - Weight decay: 0.01
- **Test Performance**:
  - Overall Accuracy: **92.5%**
  - Weighted F1-Score: **0.926**

**Per-Class Performance**:
| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Sadness  | 0.95      | 0.96   | 0.96     | 581     |
| Joy      | 0.96      | 0.93   | 0.95     | 695     |
| Love     | 0.82      | 0.89   | 0.85     | 159     |
| Anger    | 0.91      | 0.93   | 0.92     | 275     |
| Fear     | 0.94      | 0.83   | 0.88     | 224     |
| Surprise | 0.67      | 0.94   | 0.78     | 66      |

### 4. T5 Model
- Transformer-based sequence-to-sequence model
- Explores text-to-text transfer learning approach

## Results

The RoBERTa-based model achieved the best performance with **92.5% accuracy**, significantly outperforming the baseline LightGBM model (68%). The model shows strong performance across all emotion categories, with particularly high precision and recall for sadness and joy—the two most frequent classes.

**Key Observations**:
- Surprise class is the most challenging due to limited training samples (572 samples)
- High recall (0.94) but lower precision (0.67) for surprise indicates the model is sensitive to this class but sometimes over-predicts it
- Love class shows good balance between precision and recall despite being underrepresented

## Project Structure

```
Emotion_Recognition/
│
├── eda-of-the-emotion-dataset.ipynb           # Exploratory data analysis
├── emotion-recognition-bert.ipynb             # BERT/RoBERTa implementation
├── emotion-recognition-ltsm-tensorflow.ipynb  # LSTM implementation
├── emotion-recognition-t5.ipynb               # T5 implementation
└── README.md                                   # Project documentation
```

## Technologies Used

- **Platform**: Kaggle Notebooks (GPU-enabled)
- **Deep Learning Frameworks**:
  - PyTorch
  - TensorFlow
- **NLP Libraries**:
  - Transformers (Hugging Face)
  - Sentence-Transformers
- **Machine Learning**:
  - Scikit-learn
  - LightGBM
- **Data Analysis & Visualization**:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - WordCloud
  - NLTK

## Getting Started

### Prerequisites
```bash
pip install torch transformers sentence-transformers
pip install tensorflow scikit-learn lightgbm
pip install pandas numpy matplotlib seaborn wordcloud nltk
```

### Running the Notebooks

1. **Exploratory Data Analysis**:
   Open and run [eda-of-the-emotion-dataset.ipynb](eda-of-the-emotion-dataset.ipynb) to understand the dataset characteristics.

2. **Train Models**:
   - For BERT/RoBERTa: [emotion-recognition-bert.ipynb](emotion-recognition-bert.ipynb)
   - For LSTM: [emotion-recognition-ltsm-tensorflow.ipynb](emotion-recognition-ltsm-tensorflow.ipynb)
   - For T5: [emotion-recognition-t5.ipynb](emotion-recognition-t5.ipynb)

3. **Download Data**:
   Download the emotion dataset from [Kaggle](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) and adjust file paths in the notebooks accordingly.

### Note
The notebooks are configured to run on Kaggle platform with GPU support. If running locally, ensure you have appropriate hardware (GPU recommended) and adjust file paths for the dataset.

## Acknowledgments

- Dataset: [Emotion Dataset by Parul Pandey](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
- Pre-trained models from Hugging Face Model Hub
- Course: Data Mining and Knowledge Discovery
