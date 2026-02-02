# 📧 Email/SMS Spam Classifier

A machine learning project that classifies SMS and email messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques. This project covers the complete ML pipeline from data cleaning to deployment as a Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)

---

## 🎯 Project Overview

This project demonstrates an end-to-end machine learning pipeline for spam detection, achieving **97% accuracy** using Multinomial Naive Bayes classifier. The model is deployed as an interactive web application using Streamlit.

### Key Features

- **Data Processing**: Cleaned and preprocessed ~5,500 SMS messages
- **NLP Pipeline**: Tokenization, stopword removal, and stemming
- **ML Models**: Trained and compared Naive Bayes, Logistic Regression, and SVM
- **Web App**: Real-time spam classification via Streamlit interface
- **Deployment**: Model serialization with pickle for production use

---

## 🏗️ High-Level Architecture

<img width="2752" height="1536" alt="High-Level-Architecture" src="https://github.com/user-attachments/assets/4570ad85-4bc0-4a55-b50b-082db0a7ce67" />

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SMS SPAM DETECTION SYSTEM                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Raw Dataset   │───►│  Data Cleaning  │───►│  Clean Dataset  │             │
│  │   (spam.csv)    │    │  - Duplicates   │    │  (~5,500 msgs)  │             │
│  │                 │    │  - Null values  │    │                 │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Lowercase  │─►│ Tokenization│─►│  Stopword   │─►│  Stemming   │            │
│  │ Conversion  │  │   (NLTK)    │  │  Removal    │  │  (Porter)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │                    TF-IDF Vectorization                         │           │
│  │     Text ──► Term Frequency ──► Inverse Document Frequency      │           │
│  │                     ──► Numerical Feature Vector                │           │
│  └─────────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MODEL LAYER                                          │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                   │
│  │  Naive Bayes  │    │   Logistic    │    │     SVM       │                   │
│  │  (Selected)   │    │  Regression   │    │               │                   │
│  │   97% Acc     │    │               │    │               │                   │
│  └───────────────┘    └───────────────┘    └───────────────┘                   │
│           │                                                                     │
│           ▼                                                                     │
│  ┌───────────────────────────────────┐                                         │
│  │  Model Serialization (pickle)     │                                         │
│  │  - model.pkl                      │                                         │
│  │  - vectorizer.pkl                 │                                         │
│  └───────────────────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │                      Streamlit Web App                          │           │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │           │
│  │  │ User Input  │─►│  Pipeline   │─►│  Prediction Display     │ │           │
│  │  │  (Text)     │  │  Processing │  │  (Spam / Not Spam)      │ │           │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │           │
│  └─────────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │                    Streamlit Cloud                              │           │
│  │         Production-ready web application hosting                │           │
│  └─────────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Components

| Layer | Components | Description |
|-------|------------|-------------|
| **Data Layer** | spam.csv, Data Cleaning | Raw data ingestion and preprocessing |
| **Preprocessing Layer** | NLTK, PorterStemmer | Text normalization and tokenization |
| **Feature Engineering** | TF-IDF Vectorizer | Convert text to numerical features |
| **Model Layer** | Naive Bayes, Logistic Regression, SVM | ML model training and selection |
| **Application Layer** | Streamlit | Interactive web interface |
| **Deployment Layer** | Streamlit Cloud | Production hosting |

### Data Flow

```
User Input ──► Preprocessing ──► TF-IDF Vectorization ──► Model Prediction ──► Result Display
    │              │                     │                      │                   │
    │              │                     │                      │                   │
"Free prize      "free prize           [0.23, 0.45,           1 (Spam)           "⚠️ Spam"
 waiting!"        wait"                 0.12, ...]
```

---

## 📊 Exploratory Data Analysis

### Distribution of Spam vs Ham Messages

The dataset shows a significant class imbalance with ham (legitimate) messages far outnumbering spam messages.

<img width="367" height="289" alt="Picture1" src="https://github.com/user-attachments/assets/3b310280-d99f-426b-acad-8fe4b34b0ac6" />

### Message Length Distribution

Spam messages tend to be longer than legitimate messages, which serves as a useful feature for classification.

<img width="410" height="267" alt="Picture2" src="https://github.com/user-attachments/assets/6d42d9b7-c558-4e7e-b0f7-12a987c4929f" />

### Top Frequent Words in Spam Messages

Analysis of the most common words in spam messages reveals typical spam indicators like "call", "free", "txt", and "now".

<img width="508" height="308" alt="Picture3" src="https://github.com/user-attachments/assets/d6f388a5-c0b0-4d57-80bd-12eb3ac8bed8" />

---

## 🛠️ Technical Implementation

### Text Preprocessing Pipeline

The preprocessing pipeline converts raw text into clean, normalized features ready for vectorization:

<img width="525" height="103" alt="Picture4" src="https://github.com/user-attachments/assets/ad015932-65dd-439a-be2c-005dd9877aa9" />


```python
# Text Preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [ps.stem(i) for i in y if i not in stopwords.words('english')]
    return " ".join(text)
```

**Pipeline Steps:**
1. **Lowercase Conversion**: Normalize all text to lowercase
2. **Tokenization**: Split text into individual words using NLTK
3. **Alphanumeric Filtering**: Remove special characters and punctuation
4. **Stopword Removal**: Filter out common English stopwords
5. **Stemming**: Reduce words to root form using Porter Stemmer

### Streamlit Application

The web application provides a simple interface for real-time spam classification:

<img width="528" height="240" alt="Picture5" src="https://github.com/user-attachments/assets/f4fc875c-77a8-4c59-aa85-eabdb5f54c5d" />

```python
# Streamlit App Workflow
st.title('Email/SMS Spam Classifier')

input_sms = st.text_input('Enter the message')
transformed_sms = transform_text(input_sms)
vector_input = tfidf.transform([transformed_sms])
result = model.predict(vector_input)[0]

if result == 1:
    st.header("Spam")
else:
    st.header("Not Spam")
```

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 97% |
| **Precision** | High |
| **Recall** | High |

### Model Improvements Applied

- Hyperparameter tuning for optimal performance
- SMOTE for handling class imbalance
- Ensemble model exploration

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

### Requirements

```txt
streamlit
scikit-learn
nltk
pandas
numpy
pickle-mixin
```

---

## 📁 Project Structure

```
sms-spam-classifier/
│
├── app.py                      # Streamlit web application
├── Sms-Spam-Detection-Code.ipynb  # Jupyter notebook with full analysis
├── spam.csv                    # Dataset
├── model.pkl                   # Trained model (serialized)
├── vectorizer.pkl              # TF-IDF vectorizer (serialized)
├── requirements.txt            # Python dependencies
├── images/                     # Visualization images
│   ├── distribution.png
│   ├── message_length.png
│   └── frequent_words.png
└── README.md
```

---

## 🔧 How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Text    │ ──► │  Preprocessing   │ ──► │ TF-IDF Vectorize│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Spam/Not Spam  │ ◄── │    Threshold     │ ◄── │  Naive Bayes    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 📚 Technologies Used

- **Python** - Core programming language
- **NLTK** - Natural Language Processing toolkit
- **scikit-learn** - Machine learning library
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**

- GitHub: [@SUMAN0702](https://github.com/SUMAN0702)
- LinkedIn: [@sairam suman bathini](https://www.linkedin.com/in/bathini-sairam-suman/)

---

## ⭐ Acknowledgments

- Dataset sourced from UCI Machine Learning Repository
- Inspired by various NLP spam detection research papers
- Streamlit community for excellent documentation

---

<p align="center">
  Made with ❤️ for cleaner inboxes everywhere
</p>
