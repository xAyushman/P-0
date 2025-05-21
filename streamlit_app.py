import streamlit as st
import joblib
import os
import time  # For animations

# Correct Paths
model_path = os.path.join('models', 'spam_classifier.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Page Configuration
st.set_page_config(page_title="🚀 Spam Detector", layout="wide")

# Apply Custom Styling
st.markdown("""
    <style>
        body {background-color: #EEF2F7;}
        .main-container {background: white; padding: 30px; border-radius: 12px; box-shadow: 2px 2px 12px rgba(0,0,0,0.1);}
        h1 {color: #005A9C; text-align: center; margin-bottom: -10px;}
        textarea {font-size: 16px; border-radius: 8px; padding: 10px; width: 100%;}
        .stButton button {
            background-color: #005A9C; color: white; padding: 10px 20px;
            font-size: 16px; border-radius: 8px; transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #004080; transform: scale(1.05);
        }
        .report {
            padding: 15px; border-radius: 10px;
            font-size: 18px; font-weight: bold;
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            0% {opacity: 0;}
            100% {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation with Detailed Guide
st.sidebar.title("🔍 Quick Navigation")
st.sidebar.write("Follow these steps to check if your content is spam or safe:")
st.sidebar.markdown("---")

st.sidebar.subheader("📌 Steps:")
st.sidebar.write("1️⃣ **Enter your text** in the input box.")
st.sidebar.write("2️⃣ **Click 'Analyze Now'** to process your text.")
st.sidebar.write("3️⃣ The AI will classify your text as **Spam or Safe**.")
st.sidebar.write("4️⃣ If spam, take caution before clicking links or responding.")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips for Best Results:")
st.sidebar.write("✔️ Avoid using very short texts—longer messages improve accuracy.")
st.sidebar.write("✔️ If unsure about the result, try **rewording** the text.")
st.sidebar.write("✔️ Works best for **emails, messages, and promotional content**.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Example Inputs:")
st.sidebar.write("- *'Congratulations! You won a prize, claim now!'*")
st.sidebar.write("- *'Hello, let's meet for coffee tomorrow.'*")

# Main Container Without Extra White Space
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🚀 AI-Powered Spam Detector")  
st.subheader("🔎 Analyze text effortlessly!")

st.markdown("---")

# User Input Section with Better Placeholder
text_input = st.text_area("📝 **Enter your text:**", height=120, placeholder="💬 Type or paste content here...")

st.markdown("---")

# Predict Button with Dynamic Loading Animation
if st.button("🚀 Analyze Now"):
    with st.spinner("🔄 Processing... Please wait!"):
        time.sleep(2)  # Simulating a loading effect
        if text_input.strip() == "":
            st.warning("⚠️ Please enter some text first!")
        else:
            # Preprocess and predict
            transformed_input = vectorizer.transform([text_input])
            prediction = model.predict(transformed_input)[0]

            # Animated Result Display
            if prediction == 1:
                st.markdown('<div class="report" style="background: #FFCCCC; color: #C70039;">🚨 ALERT! This is **SPAM. Be cautious!**</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="report" style="background: #CCFFCC; color: #27AE60;">✅ This content is **Safe!**</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🔬 Powered by **Machine Learning** | Designed with ❤️ by **Ayushman**")
st.markdown('</div>', unsafe_allow_html=True)
