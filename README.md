# AI-Based Automotive Review and Customer Sentiment Analytics

### Aspect-Based Sentiment Analysis of Automotive Reviews using Fine-Tuned BERT

An AI-powered Natural Language Processing system for analyzing automotive customer reviews and identifying sentiment associated with specific vehicle aspects.

The project combines **Aspect-Based Sentiment Analysis (ABSA)**, **Fine-Tuned BERT**, rule-based automotive aspect extraction, and an interactive **Streamlit dashboard** to provide detailed customer sentiment insights.

---

## Live Application

### [Open AutoInsight AI — Live Project](LIVE_PROJECT_LINK)

The deployed application allows users to enter automotive customer reviews and obtain aspect-level sentiment predictions through an interactive web interface.

---

## Project Overview

Traditional sentiment analysis generally assigns a single sentiment to an entire review. Automotive reviews, however, often contain different opinions about multiple vehicle features.

For example:

> The car is expensive, but the safety features are excellent.

The proposed system can identify:

| Aspect | Sentiment |
|---|---|
| Price | Negative |
| Safety | Positive |

This enables more detailed analysis of customer opinions than conventional document-level sentiment classification.

---

## Key Features

### 1. Automotive Review Analysis

Users can enter an automotive customer review through the Streamlit interface and obtain sentiment analysis results.

### 2. Aspect Detection

The system detects automotive aspects mentioned in customer reviews.

The supported aspects are:

- Vehicle
- Engine
- Battery
- Mileage
- Safety
- Comfort
- Service
- Infotainment
- Price

### 3. Aspect-Level Sentiment Classification

Each detected aspect is independently classified into one of three sentiment categories:

- Positive
- Neutral
- Negative

### 4. Fine-Tuned BERT Model

The sentiment classifier is based on `bert-base-uncased`, fine-tuned for three-class automotive sentiment classification.

### 5. Interactive Streamlit Dashboard

The **AutoInsight AI** dashboard provides a user-friendly interface for entering reviews and viewing sentiment analysis results.

### 6. Sentiment Visualization

The application presents detected aspects, sentiment predictions, confidence information, and overall sentiment through an interactive interface.

---

## System Architecture

The project consists of two primary workflows:

### Training Pipeline

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
   Saved BERT Model
```

### Inference Pipeline

```text
Customer Review
       |
       v
Automotive Aspect Detection
       |
       v
Aspect-Specific Context
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

Detailed architecture diagrams are available in the [`docs/architecture`](./docs/architecture/) directory.

---

## Dataset

The project uses a curated and balanced automotive review dataset containing **216 labelled review sentences**.

| Dataset Property | Value |
|---|---:|
| Total Samples | 216 |
| Automotive Aspects | 9 |
| Sentiment Classes | 3 |
| Negative Samples | 72 |
| Neutral Samples | 72 |
| Positive Samples | 72 |
| Training Samples | 151 |
| Validation Samples | 32 |
| Test Samples | 33 |

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| NLP | Hugging Face Transformers |
| Transformer Model | BERT (`bert-base-uncased`) |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Dataset Handling | Hugging Face Datasets |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Development Environment | Visual Studio Code |
| Version Control | Git / GitHub |

---

## Project Structure

```text
AUTOMOTIVE-SENTIMENT-ANALYTICS/
│
├── data/
│   ├── automotive_reviews.csv
│   ├── processed_reviews.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── docs/
│   ├── architecture/
│   │   └── training_pipeline.png
│   │
│   ├── Automotive_Sentiment_Analytics_Lopamudra_Roy_Project_Report.pdf
│   ├── project-documentation.md
│   └── README.md
│
├── model/
│   └── bert_automotive_sentiment/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── notebooks/
├── outputs/
│
├── src/
│   ├── aspect_extractor.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   └── train_bert.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

> The trained BERT model weights are excluded from the GitHub repository because the model file exceeds GitHub's standard individual file-size limit.

---

## Documentation

Complete supporting documentation is available in the [`docs`](./docs/) directory.

### Technical Documentation

The detailed technical documentation covers:

- Project objectives
- Dataset design
- Data preprocessing
- System architecture
- Automotive aspect extraction
- BERT sentiment classification
- Model training
- Streamlit dashboard
- Testing and evaluation
- Limitations
- Future scope
- Execution instructions

**[View Technical Documentation](./docs/project-documentation.md)**

### Academic Project Report

The complete academic report contains the project methodology, system design, implementation, results, discussion, conclusion, references, and appendix.

**[View Complete Project Report](./docs/Automotive_Sentiment_Analytics_Lopamudra_Roy_Project_Report.pdf)**

### Architecture

System architecture and pipeline diagrams are available here:

**[View Architecture Documentation](./docs/architecture/)**

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/lopa2004/AUTOMOTIVE-SENTIMENT-ANALYTICS.git
cd AUTOMOTIVE-SENTIMENT-ANALYTICS
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Preprocess the Dataset

```bash
python src/preprocessing.py
```

### 6. Prepare Training Data

```bash
python src/prepare_data.py
```

### 7. Train the BERT Model

```bash
python src/train_bert.py
```

### 8. Run Prediction

```bash
python src/predict.py
```

### 9. Start the Streamlit Application

```bash
streamlit run app.py
```

The application normally runs locally at:

```text
http://localhost:8501
```

---

## Model Configuration

| Parameter | Configuration |
|---|---|
| Base Model | `bert-base-uncased` |
| Number of Sentiment Classes | 3 |
| Maximum Sequence Length | 128 |
| Learning Rate | 2e-5 |
| Training Batch Size | 8 |
| Evaluation Batch Size | 8 |
| Epochs | 3 |
| Weight Decay | 0.01 |
| Model Selection Metric | Weighted F1 |

---

## Limitations

- The current dataset contains 216 curated review sentences.
- Aspect extraction is based on predefined automotive keywords.
- The system currently supports nine predefined automotive aspects.
- Unseen automotive terminology may not always be detected correctly.
- Complex linguistic structures and sarcasm may affect sentiment classification.
- The current implementation focuses on English-language automotive reviews.
- Results obtained from the small curated dataset should not automatically be generalized to large real-world automotive datasets.

---

## Future Scope

Future improvements may include:

- Larger real-world automotive review datasets
- Transformer-based aspect extraction
- Improved contextual aspect detection
- Domain-specific transformer models
- Multilingual automotive review analysis
- Explainable AI for sentiment predictions
- Real-time automotive review collection
- API-based integration
- Cloud deployment
- Additional automotive aspect categories
- Advanced model evaluation and benchmarking

---

## Author

**Lopamudra Roy**

B.Tech — Computer Science & Engineering  
Institute of Engineering & Management, Kolkata  

**Enrollment Number:** 12023002001195  
**Roll Number:** 54

---

## Academic Project

**AI-Based Automotive Review and Customer Sentiment Analytics**  
*Aspect-Based Sentiment Analysis of Automotive Reviews using Fine-Tuned BERT*

Department of Computer Science and Engineering  
Institute of Engineering & Management, Kolkata