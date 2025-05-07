from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
import re

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("../sentiment_model.keras")
tokenizer = joblib.load("../sentiment_tokenizer.pkl")
max_length = 200
vocab_size = 10000
confidence_threshold = 0.6  # Require >60% confidence for prediction

def clean_text(text):
    """Clean text to match IMDB preprocessing (lowercase, remove punctuation)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Sample test reviews for reference
sample_reviews = [
    {"text": "This movie was fantastic and thrilling!", "true_label": "Positive"},
    {"text": "Awful experience, total waste of time.", "true_label": "Negative"}
]
for sample in sample_reviews:
    cleaned_text = clean_text(sample["text"])
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0][0]
    sample["predicted_label"] = "Positive" if pred > 0.5 else "Negative"
    sample["probability"] = pred if pred > 0.5 else 1 - pred

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get review text
            review = request.form["review"].strip()
            if not review:
                return render_template("index.html", error="Please enter a review", sample_reviews=sample_reviews)
            
            # Clean text
            cleaned_review = clean_text(review)
            
            # Count words
            word_count = len(cleaned_review.split())
            if word_count > max_length:
                return render_template(
                    "index.html",
                    error=f"Review must be {max_length} words or fewer",
                    sample_reviews=sample_reviews
                )

            # Preprocess text
            sequence = tokenizer.texts_to_sequences([cleaned_review])
            if not sequence or not sequence[0]:
                return render_template(
                    "index.html",
                    error="Review contains no recognized words",
                    sample_reviews=sample_reviews
                )
            padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")

            # Predict
            pred = model.predict(padded, verbose=0)[0][0]
            if pred > 0.5 and pred < confidence_threshold:
                return render_template(
                    "index.html",
                    error="Prediction confidence too low",
                    sample_reviews=sample_reviews
                )
            label = "Positive" if pred > 0.5 else "Negative"
            probability = pred if pred > 0.5 else 1 - pred

            # Debug preprocessing
            print(f"Review: {review}")
            print(f"Cleaned Review: {cleaned_review}")
            print(f"Sequence: {sequence}")
            print(f"Padded shape: {padded.shape}")
            print(f"Prediction: {label}, Probability: {pred:.4f}")

            return render_template(
                "index.html",
                result=f"{label} (confidence: {probability:.2%})",
                review=review,
                word_count=word_count,
                sample_reviews=sample_reviews
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template(
                "index.html",
                error=f"Error: {str(e)}",
                sample_reviews=sample_reviews
            )
    
    return render_template("index.html", sample_reviews=sample_reviews)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)