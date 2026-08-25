import joblib
from pathlib import Path

from src.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_spam(message):

    cleaned_message = clean_text(message)

    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    prediction = model.predict(
        message_vector
    )

    return prediction[0]

def predict_spam_details(message):

    cleaned_message = clean_text(message)

    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    prediction = model.predict(message_vector)[0]

    decision_score = float(
        model.decision_function(message_vector)[0]
    )

    return {
        "prediction": prediction,
        "decision_score": decision_score
    }




##this part is for testing

if __name__ == "__main__":

    message = "Hey, are we still meeting tomorrow at 10?"

    result = predict_spam_details(message)

    print("Message:", message)
    print("Prediction:", result["prediction"])
    print("Decision Score:", result["decision_score"])
    