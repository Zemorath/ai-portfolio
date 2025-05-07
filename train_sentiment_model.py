import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load IMDB dataset
vocab_size = 10000
max_length = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Decode word indices to text
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}
def decode_review(indices):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in indices])

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post")

# Save tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.word_index = word_index
joblib.dump(tokenizer, "sentiment_tokenizer.pkl")

# Build model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
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

# Sample predictions to verify labels
print("\nSample Predictions:")
for i in range(3):
    sample_review = X_test[i:i+1]
    sample_true = "Positive" if y_test[i] == 1 else "Negative"
    sample_pred = model.predict(sample_review)[0]
    sample_label = "Positive" if sample_pred > 0.5 else "Negative"
    print(f"Review {i+1}: {decode_review(sample_review[0][:20])}...")
    print(f"True Label: {sample_true}, Predicted: {sample_label} (probability: {sample_pred[0]:.2f})")

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

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

# Save model in Keras format
model.save("sentiment_model.keras")
print("Model and tokenizer saved.")