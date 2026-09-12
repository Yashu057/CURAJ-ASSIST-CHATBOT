# 🎓 CURAJ ASSIST — ML Admission Chatbot

An ML-based chatbot for the **Central University of Rajasthan (CURAJ)** that answers student queries related to admissions, fees, eligibility, courses, CUET, hostel facilities, and general campus information.

Built using **classical NLP and machine learning**, without deep learning or LLM APIs.

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
