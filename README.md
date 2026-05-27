# 📰 Fake News Detection Using Machine Learning

A machine learning web application that detects whether a news article is **Real or Fake** using Natural Language Processing (NLP) and multiple ML classification algorithms.

---

## 📌 Brief Description

Fake news is a growing problem on the internet and social media. This project aims to solve that by building an automated system that takes a news article as input and classifies it as **REAL ✅** or **FAKE 🚨** using trained ML models. The app is deployed using **Streamlit** for an interactive and user-friendly experience.

---

## 🛠️ Technology Stack and Tools Used

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.x |
| IDE | Spyder (Anaconda) |
| Frontend | Streamlit |
| ML Library | Scikit-learn |
| Data Handling | Pandas, NumPy |
| NLP | TF-IDF Vectorizer |
| Model Saving | Joblib |
| Dataset | Kaggle — Fake and Real News Dataset |

---

## ✨ Features and Functionalities Implemented

- Loads and merges **44,000+ real and fake news articles** from Kaggle
- Text **preprocessing** — lowercasing, removing punctuation and special characters
- **TF-IDF Vectorization** — converts raw text into numerical features (5000 max features)
- Trains and compares **4 ML models**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Naive Bayes (Multinomial)
- Prints **accuracy, precision, recall, and F1-score** for all models
- Automatically selects and **saves the best model** using joblib
- **Streamlit web app** where user can paste any news article and get instant prediction
- Shows **confidence percentage** for Real vs Fake

---

## ⚙️ Installation / Execution Steps to Run the Project

### Prerequisites
Make sure you have Python installed (Anaconda recommended).

### Step 1 — Clone or download this repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### Step 2 — Install required libraries
```bash
pip install pandas numpy scikit-learn streamlit joblib
```

### Step 3 — Download the dataset
- Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Download `Fake.csv` and `True.csv`
- Place both files in the project folder

### Step 4 — Train the model (run in Spyder or terminal)
```bash
python model.py
```
This will:
- Train all 4 models
- Print accuracy results
- Save `best_model.pkl` and `tfidf.pkl` in the project folder

### Step 5 — Run the Streamlit app
```bash
streamlit run app.py
```
- Browser will open automatically
- Paste any news article in the text box
- Click **Detect** to see the result

---

## 📁 Project Structure

```
fake-news-detection/
│
├── Fake.csv               # Fake news dataset
├── True.csv               # Real news dataset
├── model.py               # ML training script
├── app.py                 # Streamlit web app
├── best_model.pkl         # Saved best model (auto-generated)
├── tfidf.pkl              # Saved TF-IDF vectorizer (auto-generated)
├── README.md              # Project documentation
├── report/
│   └── project_report.pdf # Detailed project report
└── screenshots/
    ├── app_real.png        # Screenshot — real news result
    └── app_fake.png        # Screenshot — fake news result
```

---

## 📊 Model Accuracy Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~98% |
| Decision Tree | ~99% |
| Random Forest | ~99% |
| Naive Bayes | ~94% |

Best performing model is automatically selected and saved.

---



---

## 👥 Team Members

| Name | Roll Number |
|---|---|
| Anant Joshi | *(add roll no.)* |
| *(ANSH GUPTA)* | *(EN23CS301149)* |
| *(ANAY DESAI)* | *(EN23CS301128)* |

**Institution:** Medi-Caps University, Indore
**Batch:** 2023–2027
**Subject:** *(MINI PROJECT)*

---

## 📄 License

This project is for academic purposes only.
