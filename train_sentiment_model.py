import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load IMDB dataset
vocab_size = 10000  # Limit to top 10,000 words
max_length = 200    # Max review length (words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Decode word indices to text (for understanding)
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}
def decode_review(indices):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in indices])
print("Sample review:", decode_review(X_train[0][:10]), "...")

# Pad sequences to fixed length
X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post")

# Save tokenizer for Flask
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.word_index = word_index
joblib.dump(tokenizer, "sentiment_tokenizer.pkl")

# Build model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),  # Convert words to vectors
    LSTM(64, return_sequences=False),                    # Process sequential text
    Dropout(0.5),                                       # Prevent overfitting
    Dense(32, activation="relu"),                       # Hidden layer
    Dense(1, activation="sigmoid")                      # Output: 0 (negative) or 1 (positive)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test) > 0.5).astype(int)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Confusion Matrix:\n{cm}")

# Visualize training progress
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_progress.png")
plt.close()

# Save model
model.save("sentiment_model.h5")
print("Model and tokenizer saved.")