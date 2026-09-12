# 🎓 CURAJ ASSIST — ML Admission Chatbot

An ML-based chatbot for the **Central University of Rajasthan (CURAJ)** that answers student queries related to admissions, fees, eligibility, courses, CUET, hostel facilities, and general campus information.

Built using **classical Natural Language Processing (NLP) and Machine Learning**, without deep learning or LLM APIs.

## 📌 Overview

Students often need to search multiple university documents and webpages to find admission and academic information. CURAJ ASSIST converts common student questions into classified intents and returns the corresponding response through a Streamlit chat interface.

### Core Approach

- **Text Representation:** TF-IDF with unigrams and bigrams
- **Classifier:** Multinomial Logistic Regression
- **Low-Confidence Fallback:** Cosine Similarity
- **Interface:** Streamlit
- **Training Data:** Intent-based JSON dataset with 40+ categories

## 🧠 ML Pipeline

```text
Student Query
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Confidence Check
   ↙       ↘
High       Low
 ↓          ↓
Response   Cosine Similarity
              ↓
           Response
```

### Preprocessing & Features

- Lowercasing and special-character handling
- Lightweight custom stopword filtering
- Suffix-based lemmatization
- TF-IDF unigrams + bigrams
- Sublinear TF scaling
- Maximum vocabulary of 5,000 features

### Training & Evaluation

- Stratified 80/20 train-test split
- Multinomial Logistic Regression with regularization
- Test-set evaluation
- Cross-validation
- Per-class precision, recall and F1-score
- Confusion matrix visualization

## 💬 Supported Queries

| Area | Example |
|---|---|
| Admissions | How do I apply to CURAJ? |
| Fees | What is the PG fee structure? |
| Eligibility | What is the eligibility for M.Sc. Computer Science? |
| Courses | Which UG/PG programmes are available? |
| CUET | What are the CUET UG/PG paper codes? |
| Hostel | What hostel facilities are available? |
| General | Tell me about CURAJ / contact information |

## 📁 Project Structure

```text
CURAJ-ASSIST-CHATBOT/
├── app.py                 # Streamlit application
├── model.ipynb            # Training and evaluation notebook
├── dataset.json           # Intent dataset
├── model.pkl              # Trained Logistic Regression model
├── vectorizer.pkl         # Fitted TF-IDF vectorizer
├── confusion_matrix.png   # Evaluation visualization
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Run Locally

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Yashu057/CURAJ-ASSIST-CHATBOT.git
cd CURAJ-ASSIST-CHATBOT
pip install -r requirements.txt
```

### Launch the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

The pre-trained model and vectorizer are included, so retraining is optional. To retrain or inspect the complete pipeline, open `model.ipynb`.

## 🛠️ Tech Stack

**Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn · Jupyter Notebook · Git**

## 📚 Data Sources

The intent dataset was prepared using information from official CURAJ documents and university resources, including fee structures, programme eligibility, and CUET mappings.

Official university website:

https://www.curaj.ac.in/

## ⚠️ Academic Scope

This is an academic/student project demonstrating **NLP, intent classification, and classical machine learning**.

University information can change over time. Important admission, eligibility, and fee details should always be verified against the latest official CURAJ notifications.

## 👤 Author

**Yash Verma**

M.Sc. Computer Science (Big Data Analytics)  
Central University of Rajasthan

## 📜 License

For educational and academic purposes.
