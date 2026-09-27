import joblib

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


print("================================")
print("      FAKE NEWS DETECTOR")
print("================================")

# Get news from user
news = input("\nEnter the news article:\n")

# Convert news into TF-IDF
news_tfidf = vectorizer.transform([news])

# Predict
prediction = model.predict(news_tfidf)[0]

# Display result
if prediction == 0:
    print("\nResult: FAKE NEWS ❌")
else:
    print("\nResult: REAL NEWS ✅")