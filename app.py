from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"].strip()

    # Check for empty or very short input
    if len(news) < 20:
        return render_template(
            "index.html",
            error="Please enter a longer news article."
        )

    # Convert news into TF-IDF
    news_tfidf = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(news_tfidf)[0]

    # Probability
    probabilities = model.predict_proba(news_tfidf)[0]
    probability = max(probabilities) * 100

    if prediction == 0:
        result = "FAKE NEWS ❌"
    else:
        result = "REAL NEWS ✅"

    # Confidence level
    if probability >= 80:
        confidence = "High Confidence"
    elif probability >= 60:
        confidence = "Medium Confidence"
    else:
        confidence = "Low Confidence"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 2),
        confidence=confidence,
        news=news
    )


if __name__ == "__main__":
    app.run(debug=True)