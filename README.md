# 🎓 CURAJ Admission Chatbot

An intelligent ML-based chatbot for the **Central University of Rajasthan (CURAJ)** that answers student queries about admissions, fees, eligibility, courses, CUET requirements, and campus facilities.

Built as a CIA assessment project for PG Data Science.

---

## 📌 Project Overview

Students applying to CURAJ often need to search through multiple PDFs and web pages for information about admissions, fee structures, eligibility criteria, and CUET subject codes. This chatbot simplifies that process by providing instant answers.

### ML Approach
- **Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency)
- **Classification Model:** Logistic Regression
- **Fallback:** Cosine Similarity for low-confidence predictions
- **No deep learning or LLMs used** — purely classical ML

---

## 📁 Project Structure

```
curaj-chatbot/
├── model.ipynb          # Training notebook (full ML pipeline)
├── app.py               # Streamlit chatbot application
├── dataset.json         # Intent-based training dataset
├── model.pkl            # Trained Logistic Regression model
├── vectorizer.pkl       # Fitted TF-IDF vectorizer
├── requirements.txt     # Python dependencies
├── confusion_matrix.png # Model evaluation visualization (generated after training)
└── README.md            # This file
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/curaj-chatbot.git
   cd curaj-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional — pre-trained model is included)
   ```bash
   jupyter notebook model.ipynb
   # Run all cells in order
   ```

4. **Launch the chatbot**
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`

---

## 🧠 ML Pipeline

### 1. Problem Definition
Intent classification — predict what the student is asking about and return the right answer.

### 2. Data Collection
- Extracted text from official CURAJ PDFs (UG/PG fee structure, CUET subject codes, eligibility documents)
- Created structured intent dataset with patterns and responses
- 40+ intent categories covering all major student queries

### 3. Data Preprocessing
- Lowercasing and special character removal
- Custom lightweight stopword list and suffix-based lemmatizer (no heavy NLP library needed)
- Custom stopword filtering (kept question words like "what", "how")

### 4. Feature Engineering
- TF-IDF vectorization with unigrams + bigrams
- Sublinear TF scaling for better term weighting
- 5000 max features to control vocabulary size

### 5. Model Training
- Logistic Regression with multinomial classification
- Stratified train-test split (80/20)
- Regularization parameter C=10

### 6. Evaluation
- Test set accuracy and cross-validation scores
- Per-class precision, recall, and F1-score
- Confusion matrix visualization

### 7. Deployment
- Streamlit web application with chat interface
- Confidence-based prediction with cosine similarity fallback

---

## 💬 What the Chatbot Can Answer

| Category | Example Questions |
|----------|-------------------|
| **Fees** | "What is the PG fee structure?", "How much is hostel fee?" |
| **Eligibility** | "Eligibility for MSc Computer Science?", "Who can apply for MBA?" |
| **Courses** | "What UG courses are available?", "List of PG programmes" |
| **CUET** | "CUET PG paper codes?", "What subjects for CUET UG?" |
| **Admissions** | "How to apply to CURAJ?", "Admission process?" |
| **Hostel** | "Hostel facilities?", "Mess charges?" |
| **General** | "Tell me about CURAJ", "Contact information" |

---

## 📊 Data Sources

- **UG Fee Structure PDF** — Fee Structure for UG programmes 2025-26
- **PG Fee Structure PDF** — Fee Structure for PG programmes 2025-26
- **CUET UG Mapping PDF** — CUET (UG) subject codes and eligibility
- **CUET PG Mapping PDF** — CUET (PG) paper codes and eligibility
- **CURAJ Website** — https://www.curaj.ac.in/

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML | scikit-learn |
| Web App | Streamlit |
| Data | JSON, pandas |
| Visualization | matplotlib, seaborn |
| Language | Python 3 |

---

## 📝 Notes

- The model uses a confidence threshold (0.35) — queries below this threshold trigger the cosine similarity fallback
- All data is sourced from official CURAJ documents for accuracy
- The chatbot works offline once the model is trained (no API calls needed)
- This is a student project using simple ML techniques — not a production system

---

## 👤 Author

PG Data Science Student, Central University of Rajasthan

---

## 📜 License

This project is for educational/academic purposes only.
