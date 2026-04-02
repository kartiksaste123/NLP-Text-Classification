# NLP Preprocessing and Text Classification

## 📌 Project Description
This project implements Natural Language Processing (NLP) techniques and builds a text classification model using machine learning algorithms.

The system processes raw text data, converts it into numerical form, and classifies messages as Spam or Not Spam.

## 🎯 Objectives
- Apply NLP preprocessing techniques
- Perform text vectorization using TF-IDF
- Build machine learning classification models
- Evaluate model performance using metrics

## 📂 Dataset
The project uses the SMS Spam Collection Dataset.

- Total messages: ~5572  
- Classes:
  - Spam (1)
  - Ham / Not Spam (0)

## ⚙️ Technologies Used
- Python
- NLTK
- Scikit-learn
- Pandas
- NumPy

## 🔧 NLP Preprocessing Steps
- Tokenization
- Stopword Removal
- Stemming
- Lemmatization
- Text Cleaning

## 🔢 Feature Extraction
- TF-IDF

## 🤖 Models Used
1. Naive Bayes  
2. Logistic Regression  

## 📊 Results

| Model | Accuracy |
|------|--------|
| Naive Bayes | 96.9% |
| Logistic Regression | 95.5% |

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

## 🔍 Observations
- Naive Bayes performed better than Logistic Regression
- High accuracy achieved
- Some spam messages are missed

## 🧪 Sample Prediction
Input:
Congratulations! You won a free lottery ticket

Output:
Not Spam

## ⚠️ Limitations
- Misclassification of some spam messages
- Limited contextual understanding

## 🚀 Future Work
- Use deep learning models (LSTM, BERT)
- Improve dataset balance

## 📁 Project Structure
NLP/
├── NLP.py
├── README.md

## 📌 Conclusion
This project demonstrates NLP preprocessing and text classification using machine learning.

## 👨‍💻 Author
Kartik Saste
"@ | Out-File -Encoding utf8 README.md
