import streamlit as st
import joblib
import os

# Correct Paths
model_path = os.path.join('models', 'spam_classifier.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Streamlit app setup
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Email Spam Classifier")
st.markdown("---")
st.write("🔍 Enter the content of an email and the app will predict whether it's **Spam** or **Not Spam (Ham)**.")

# Text input
email_input = st.text_area("✉️ Enter your email content here:")

# Predict Button
if st.button("🚀 Predict"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some email text first!")
    else:
        # Preprocess and predict
        transformed_input = vectorizer.transform([email_input])
        prediction = model.predict(transformed_input)[0]

        # Show prediction result
        if prediction == 1:
            st.error("🚨 ALERT! This email is predicted to be **SPAM**.")
        else:
            st.success("✅ Great! This email is predicted to be **NOT SPAM** (Ham).")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Machine Learning.")

