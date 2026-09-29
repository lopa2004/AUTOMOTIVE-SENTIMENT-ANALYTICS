# Project Documentation

## AI-Based Automotive Review and Customer Sentiment Analytics

### Aspect-Based Sentiment Analysis of Automotive Reviews Using Fine-Tuned BERT

**Author:** Lopamudra Roy  
**Programme:** B.Tech in Computer Science and Engineering  
**Institute:** Institute of Engineering & Management, Kolkata  
**Enrollment Number:** 12023002001195  
**Roll Number:** 54  

---

## 1. Project Overview

This project presents an AI-based system for analyzing customer sentiment in automotive reviews using Aspect-Based Sentiment Analysis (ABSA) and a fine-tuned BERT model.

Unlike traditional sentiment analysis, which assigns a single sentiment to an entire review, the proposed system identifies specific automotive aspects mentioned in a review and determines the sentiment associated with each aspect.

The system currently supports the following automotive aspects:

- Vehicle
- Engine
- Battery
- Mileage
- Safety
- Comfort
- Service
- Infotainment
- Price

Each detected aspect is classified into one of three sentiment categories:

- Positive
- Neutral
- Negative

The complete system is integrated into an interactive Streamlit application named **AutoInsight AI**.

---

## 2. Live Application

The deployed version of the project can be accessed using the following link:

**[Open AutoInsight AI — Live Application](LIVE_LINK_HERE)**

The application allows users to enter automotive customer reviews and obtain aspect-level sentiment analysis results through an interactive web interface.

---

## 3. Project Report

The complete academic project report contains the project background, objectives, methodology, system design, implementation, model training, testing, results, limitations, and future scope.

**[View Complete Project Report](./Automotive_Sentiment_Analytics_Lopamudra_Roy_Project_Report.pdf)**

---

## 4. System Architecture

The proposed system consists of two primary workflows:

1. Model Training Pipeline
2. Sentiment Inference Pipeline

The overall architecture combines data preprocessing, automotive aspect identification, BERT-based sentiment classification, and a Streamlit-based user interface.

---

## 5. Model Training Pipeline

The training pipeline is responsible for preparing the automotive review dataset and fine-tuning the BERT sentiment classification model.

### Training Workflow

```text
Automotive Review Dataset
          |
          v
   Data Preprocessing
          |
          v
    Dataset Preparation
          |
          v
Train / Validation / Test Split
          |
          v
    BERT Tokenization
          |
          v
   Fine-Tuning BERT
          |
          v
 Model Evaluation
          |
          v
 Saved Trained Model
```

### Training Pipeline Diagram

![Training Pipeline](./architecture/training_pipeline.png)

The trained model and tokenizer configuration are subsequently used by the sentiment inference system.

---

## 6. Sentiment Inference Pipeline

The inference pipeline processes a customer review submitted through the application and generates aspect-level sentiment predictions.

### Inference Workflow

```text
Customer Review
      |
      v
Automotive Aspect Detection
      |
      v
Aspect-Specific Context Extraction
      |
      v
Fine-Tuned BERT Model
      |
      v
Sentiment Classification
      |
      v
Positive / Neutral / Negative
      |
      v
AutoInsight AI Dashboard
```

The final results are presented through the Streamlit interface in a structured and user-friendly format.

---

## 7. Dataset

The project uses a curated automotive sentiment dataset containing **216 labelled review sentences**.

| Dataset Property | Value |
|---|---:|
| Total Samples | 216 |
| Automotive Aspects | 9 |
| Sentiment Classes | 3 |
| Training Samples | 151 |
| Validation Samples | 32 |
| Test Samples | 33 |

The dataset contains balanced sentiment labels for Positive, Neutral, and Negative sentiment categories.

---

## 8. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Transformer Model | BERT (`bert-base-uncased`) |
| NLP Framework | Hugging Face Transformers |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Dataset Handling | Hugging Face Datasets |
| Web Application | Streamlit |
| Visualization | Plotly |
| Development Environment | Visual Studio Code |
| Version Control | Git and GitHub |

---

## 9. Project Documentation

Detailed technical documentation covering the design and implementation of the system is available below:

**[View Technical Project Documentation](./project-documentation.md)**

The documentation includes:

- Project objectives
- Dataset design
- Data preprocessing
- Aspect extraction
- BERT model architecture
- Model training
- Sentiment prediction
- Streamlit dashboard
- Testing and evaluation
- Limitations
- Future improvements
- Execution instructions

---

## 10. Repository Navigation

Return to the main repository documentation:

**[Back to Main README](../README.md)**

---

## Author

**Lopamudra Roy**  
B.Tech in Computer Science and Engineering  
Institute of Engineering & Management, Kolkata  
Enrollment Number: 12023002001195  
Roll Number: 54