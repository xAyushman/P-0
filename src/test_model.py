import joblib

# Load the trained model and vectorizer
model = joblib.load('../models/spam_classifier.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# Function to predict if a message is spam or not
def predict_spam(email_text):
    email_tfidf = vectorizer.transform([email_text])  # Transform input text
    prediction = model.predict(email_tfidf)  # Predict spam (1) or ham (0)
    return "Spam" if prediction[0] == 1 else "Ham"

# Live testing loop
print("\n--- Spam Detector Live Testing ---")
print("Type your email message below to check if it's spam or not.")
print("Type 'exit' to stop.\n")

while True:
    email_text = input("Enter an email message: ")
    if email_text.lower() == 'exit':
        print("\nExiting Spam Detector. Goodbye! 👋")
        break
    print(f"Prediction: {predict_spam(email_text)}\n")
