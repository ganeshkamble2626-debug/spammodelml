import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("scaled.joblib")
    return model, vectorizer

model, vectorizer = load_model()

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}
h1 {
    text-align: center;
    color: #333;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 25px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}
.spam {
    background-color: #ffe6e6;
    color: #d63031;
}
.ham {
    background-color: #eaffea;
    color: #2d8a34;
}
</style>
""", unsafe_allow_html=True)

# ------------------- UI -------------------
st.title("📩 Spam Message Classifier")
st.write("### Enter a message to check whether it is **Spam** or **Ham**")

message = st.text_area(
    "✉️ Type your message here:",
    height=120,
    placeholder="Congratulations! You won a free prize..."
)

# ------------------- PREDICTION -------------------
if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Transform input
        message_vec = vectorizer.transform([message])

        # Predict label
        prediction = model.predict(message_vec)[0]

        # Predict probability (if supported)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(message_vec)[0]
            spam_prob = proba[1] * 100
            ham_prob = proba[0] * 100
        else:
            spam_prob = ham_prob = None

        # ------------------- RESULT -------------------
        if prediction == 1:
            st.markdown(
                f"<div class='result-box spam'>🚨 SPAM MESSAGE</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box ham'>✅ HAM (Not Spam)</div>",
                unsafe_allow_html=True
            )

        # ------------------- PROBABILITY GRAPH -------------------
        if spam_prob is not None:
            st.write("### 📊 Prediction Confidence")

            labels = ["Ham", "Spam"]
            values = [ham_prob, spam_prob]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Probability (%)")
            ax.set_title("Spam vs Ham Probability")

            for i, v in enumerate(values):
                ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

            st.pyplot(fig)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown(
    "<center>🤖 Built with Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)