import streamlit as st
import joblib

# Charger le modèle et le vectoriseur
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('random_forest_model.joblib')


# Fonction de prédiction
def predict_sentiment(text):
    # Transformer le texte avec le vectoriseur
    text_transformed = vectorizer.transform([text])
    # Prédire avec le modèle
    prediction = model.predict(text_transformed)
    return prediction[0]

# Interface Streamlit
st.title("Sentiment Analysis Predictor")

user_input = st.text_area("Kteb commentaire b l'anglais ta3rav kano positive walla negative kivtak")

if st.button("Predict"):
    if user_input:
        result = predict_sentiment(user_input)
        st.write(f"The predicted sentiment is: **{result}**")
    else:
        st.write("Please enter some text for prediction.")

