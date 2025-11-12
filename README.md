# 💊 Medicine Review Sentiment Analysis

## 🧠 Project Overview

This project performs **Sentiment Analysis on Drug Reviews** to determine whether a user’s review about a medicine is **Positive** or **Negative**.
It combines **Natural Language Processing (NLP)** and **Machine Learning** to build a predictive model and a **Streamlit web application** for real-time sentiment prediction.

---

## 📁 Repository Structure

```
📦 Drug-Review-Sentiment-Analysis
│
├── 📜 app.py                        # Streamlit web app for real-time sentiment prediction
├── 📔 Drug_Review_Sentiment_Analysis.ipynb  # Model training and evaluation notebook
├── 📄 drugsComTest_raw.csv          # Dataset containing drug reviews and ratings
├── 📦 logistic_regression_sentiment_model.pkl  # Trained Logistic Regression model
├── 📦 tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── 🎞️ animation.json                # Optional Lottie animation for UI
└── 📝 README.md                     # Project documentation (this file)
```

---

## ⚙️ Key Features

* 🔤 **Text Preprocessing** — Cleans and tokenizes user input by removing stopwords, punctuation, and special characters.
* 🧩 **Machine Learning Model** — A Logistic Regression classifier trained on drug review data to predict sentiment.
* 📊 **TF-IDF Vectorization** — Transforms text into numerical vectors for model interpretation.
* 🧬 **Streamlit Web App** — Simple, clean, and interactive interface to analyze user reviews instantly.
* 💡 **Confidence Score** — Displays prediction confidence for better interpretability.
* 🎨 **Lottie Animation Support** — Adds an engaging visual animation to the Streamlit interface.

---

## 🧰 Technologies Used

| Category             | Tools / Libraries                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------ |
| Programming Language | Python                                                                                     |
| Data Handling        | Pandas, NumPy                                                                              |
| NLP                  | NLTK, re (Regex), TF-IDF                                                                   |
| Machine Learning     | Scikit-learn                                                                               |
| Model Persistence    | joblib                                                                                     |
| Web App              | Streamlit                                                                                  |
| Visualization        | Streamlit-Lottie                                                                           |
| Dataset              | [drugsComTest_raw.csv](https://www.kaggle.com/datasets/jessicali9530/drug-reviews-dataset) |

---

## 🧪 Model Workflow

1. **Dataset Loading:**

   * Uses `drugsComTest_raw.csv` which contains medicine reviews, ratings, and conditions.
2. **Data Cleaning & Preprocessing:**

   * Lowercasing, punctuation removal, stopword elimination, and lemmatization (if applied).
3. **Feature Extraction:**

   * TF-IDF vectorization converts text reviews into numerical features.
4. **Model Training:**

   * A Logistic Regression classifier is trained to classify reviews as Positive or Negative.
5. **Model Evaluation:**

   * Evaluated using metrics like Accuracy, Precision, Recall, and F1-Score.
6. **Deployment:**

   * The trained model and TF-IDF vectorizer are saved using `joblib`.
   * Deployed using Streamlit (`app.py`) for real-time predictions.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Drug-Review-Sentiment-Analysis.git
cd Drug-Review-Sentiment-Analysis
```

### 2️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

> Example dependencies (if you want to create a `requirements.txt`):

```
streamlit
pandas
numpy
scikit-learn
joblib
requests
streamlit-lottie
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

### 4️⃣ Enter a Medicine Review

Type any drug review in the text box (e.g., *“This medicine really helped reduce my pain!”*) and click **Analyze**.
You’ll get an instant prediction along with the confidence level.

---

## 📈 Example Output

**Input:**

> “This medicine worked great for my headache!”

**Output:**
✅ **Predicted Sentiment:** Positive (96.3%)

---

## 🧩 Future Improvements

* Integrate deep learning models (LSTM, BERT, etc.)
* Add neutral sentiment classification
* Improve data preprocessing pipeline
* Add multi-language support
* Deploy on cloud (Streamlit Cloud / Hugging Face Spaces / Render)

---

## 👨‍💻 Author

**Lokesh Sohanda**
🎵 Data Science Enthusiast | AI & ML Learner | Music Composer

---
