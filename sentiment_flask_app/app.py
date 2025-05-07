from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("../sentiment_model.keras")
tokenizer = joblib.load("../sentiment_tokenizer.pkl")
max_length = 200
vocab_size = 10000

# Sample test reviews for reference
word_index = joblib.load("../sentiment_tokenizer.pkl").word_index
reverse_word_index = {v: k for k, v in word_index.items()}
def decode_review(indices):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in indices])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get review text
            review = request.form["review"].strip()
            if not review:
                return render_template("index.html", error="Please enter a review")
            if len(review.split()) > max_length:
                return render_template("index.html", error=f"Review must be {max_length} words or fewer")

            # Preprocess text
            sequence = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")

            # Predict
            pred = model.predict(padded, verbose=0)[0][0]
            label = "Positive" if pred > 0.5 else "Negative"
            probability = pred if pred > 0.5 else 1 - pred

            return render_template(
                "index.html",
                result=f"{label} (confidence: {probability:.2%})",
                review=review
            )
        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")
    
    # Load sample test reviews
    sample_reviews = [
        {"text": "This movie was fantastic and thrilling!", "true_label": "Positive"},
        {"text": "Awful experience, total waste of time.", "true_label": "Negative"}
    ]
    for sample in sample_reviews:
        sequence = tokenizer.texts_to_sequences([sample["text"]])
        padded = pad_sequences(sequence, maxlen=max_length, padding="post")
        pred = model.predict(padded, verbose=0)[0][0]
        sample["predicted_label"] = "Positive" if pred > 0.5 else "Negative"
        sample["probability"] = pred if pred > 0.5 else 1 - pred

    return render_template("index.html", sample_reviews=sample_reviews)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)