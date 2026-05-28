# 📰 Fake News Detection Using Machine Learning

A machine learning web application that detects whether a news article is **Real or Fake** using Natural Language Processing (NLP) and ML classification algorithms.

📌 📌 📌 LIVE DEMO : https://fake-news-detector-truthlens.streamlit.app/ 
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
| Model Saving | Pickle |
| Dataset | Kaggle — Fake and Real News Dataset |

---

## ✨ Features and Functionalities Implemented

- Loads and merges **44,000+ real and fake news articles** from Kaggle
- Text **preprocessing** — lowercasing, removing punctuation and special characters
- **TF-IDF Vectorization** — converts raw text into numerical features (5000 max features)
- Trains and ** model**:
  - Logistic Regression
  
- Prints **accuracy, precision, recall, and F1-score** for all models
- Automatically selects and **saves the model** using pickle
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
pip install pandas numpy scikit-learn streamlit pickle
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
- Trains model
- Print accuracy results
- Save `models.pkl` and `tfidf.pkl` in the project folder

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


#SCREENSHOTS:

<img width="1312" height="983" alt="Screenshot 2026-05-28 at 12 35 51 AM" src="https://github.com/user-attachments/assets/7a841e50-6aee-4c1b-8f6a-fd8bffa13876" />
**<img width="3420" height="2050" alt="image" src="https://github.com/user-attachments/assets/19a8fe5f-ea84-4584-87ac-999daa419657" />
**

---

## 👥 Team Members

| Name | Roll Number |
|---|---|
| ANANT JOSHI| EN23CS301125|
| ANAY DESAI | EN23CS301128 |
| ANSH GUPTA | EN23CS301149 |

**Institution:** Medi-Caps University, Indore
**Batch:** 2023–2027
**Subject:** *(MINI PROJECT)*

---

## 📄 License

This project is for academic purposes only.
