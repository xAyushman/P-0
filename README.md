# 📧 Email Spam Detector

## 🚀 Overview
The **Email Spam Detector** is a machine learning-based system designed to classify emails as either **spam** or **ham (not spam)**. It uses **Natural Language Processing (NLP)** techniques and a **Naïve Bayes classifier** to achieve high accuracy in spam detection.

## 📂 Project Structure
```
P-0/
│-- data/                  # Contains datasets
│   │-- spam.csv           # Original dataset
│   │-- merged_spam_dataset.csv # Merged dataset (after adding more data)
│-- models/                # Saved models and vectorizers
│   │-- spam_classifier.pkl
│   │-- tfidf_vectorizer.pkl
│-- src/                   # Source code
│   │-- train.py           # Training script
│   │-- test.py            # Testing script
│   │-- predict.py         # Live testing script
│-- README.md              # Project documentation
```

## 🔬 Features
✅ **Spam Detection** - Classifies emails as spam or ham using NLP & ML.  
✅ **TF-IDF Vectorization** - Extracts features from text.  
✅ **SMOTE** - Handles class imbalance in the dataset.  
✅ **Hyperparameter Tuning** - Optimizes the model’s `alpha` value.  
✅ **Custom Thresholding** - Allows adjusting the spam detection sensitivity.  
✅ **Live Testing** - Users can input an email to check if it’s spam.  

## 📊 Model Performance
| Metric      | Score |
|------------|-------|
| Accuracy   | 98.50%|
| Precision  | 99.57%|
| Recall     | 97.00%|

## 🛠️ Installation
1️⃣ **Clone the Repository:**  
```sh
git clone https://github.com/your-username/email-spam-detector.git
cd email-spam-detector
```
2️⃣ **Install Dependencies:**  
```sh
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Model
To train the model with the dataset:
```sh
python src/train.py
```
The trained model and vectorizer will be saved in the `models/` folder.

## 🧪 Testing the Model
Run the `test.py` script to evaluate the model’s performance:
```sh
python src/test.py
```

## 📝 Live Testing with User Input
Test a custom email using:
```sh
python src/predict.py
```
Enter your email text, and the model will predict if it's spam or ham.

## 📌 Future Enhancements
- Improve dataset by adding more labeled spam emails.
- Experiment with deep learning models (LSTMs, Transformers).
- Deploy as a web application using Flask or FastAPI.

## 💡 Contributing
If you'd like to contribute, feel free to fork this repository and submit a pull request!

## 📜 License
This project is licensed under the **MIT License**.

---
Made with ❤️ by [xAyushman]

