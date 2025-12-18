import streamlit as st
import joblib
# load model and vectorizer
model=joblib.load("model.joblib")
vectorizer=joblib.load("scaled.joblib")

#Set page title
st.set_page_config(page_title="Spam Detector",layout="centered")

st.title("Spam Message Classifier")
st.write("Enter a message below to check if its **spam** or **ham**.")

#Text input
message=st.text_area("Type your message:",height=150)

#predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        #transform message using loaded vectorizer
        X_input=vectorizer.transform([message])
        
        #make prediction
        prediction=model.predict(X_input)[0]
        
        #output result
        if prediction == "Spam":
            st.error("This message is **SPAM**.")
        else:
            st.success("This message is **HAM** (Not Spam)")