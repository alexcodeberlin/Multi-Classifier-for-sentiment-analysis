import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import csv
import numpy as np
import matplotlib.pyplot as plt

# Define list of all labels
emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", 
            "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", 
            "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", 
            "relief", "remorse", "sadness", "surprise", "neutral"]

# Initialize lists for sentences and labels
sentences = []
labels = []

# Load training data
for filename in ["goemotions_1.csv", "goemotions_2.csv"]:
    try:
        with open(f"C:/Users/alexa/Desktop/New folder (3)/full_dataset/{filename}", "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row["text"]
                sentences.append(text)
                for emotion in emotions:
                    if int(row[emotion]) == 1:
                        labels.append(emotion)
                        break
    except FileNotFoundError:
        print(f"File {filename} not found. Please check the path.")

# Load validation data
val_sentences = []
val_labels = []
try:
    with open("C:/Users/alexa/Desktop/New folder (3)/full_dataset/goemotions_3.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row["text"]
            val_sentences.append(text)
            for emotion in emotions:
                if int(row[emotion]) == 1:
                    val_labels.append(emotion)
                    break
except FileNotFoundError:
    print("Validation data file not found. Please check the path.")

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
val_labels = encoder.transform(val_labels)  # Use transform instead of fit_transform on validation set
labels = to_categorical(labels, num_classes=len(emotions))
val_labels = to_categorical(val_labels, num_classes=len(emotions))

# Tokenize sentences
tokenizer = Tokenizer(oov_token='oov')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convert sentences to sequences and pad them
sequences = tokenizer.texts_to_sequences(sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
padded = pad_sequences(sequences, padding="post")
val_padded = pad_sequences(val_sequences, padding="post")

# Debugging output
print("Number of training samples:", len(padded))
print("Number of validation samples:", len(val_padded))
print("Shape of padded input data:", padded.shape)
print("Shape of labels data:", labels.shape)

# Ensure input and output sizes match
min_length = min(len(padded), len(labels))
padded = padded[:min_length]
labels = labels[:min_length]

# Randomly sample to ensure equal size (if necessary)
if len(padded) != len(labels):
    print("Randomly sampling input data to match the number of labels.")
    indices = np.random.choice(len(padded), len(labels), replace=False)
    padded = padded[indices]

# Model definition
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=150, input_length=padded.shape[1]))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Bidirectional(LSTM(100)))
model.add(BatchNormalization())
model.add(Dense(units=len(emotions), activation='softmax'))

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_accuracy')
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True, min_delta=0.1)
val_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00005, mode='min')

# Fit model with user-defined epochs and batch size
epochs = 1  # You can change this value
batch_size = 64  # You can change this value

history = model.fit(
    padded, labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_padded, val_labels),
    callbacks=[checkpoint, early_stop, val_reduce]
)

# Save model weights
model.save_weights("nlp_emotion_weights.h5", overwrite=True)

# Output final training and validation accuracy
print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Allow user to input a sentence for emotion prediction
def predict_emotion(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=padded.shape[1], padding='post')
    prediction = model.predict(padded_sequence)
    emotion = encoder.inverse_transform([np.argmax(prediction)])
    return emotion[0]

# User input for emotion prediction
user_sentence = input("Enter a sentence to predict its emotional state: ")
predicted_emotion = predict_emotion(user_sentence)
print(f"The predicted emotional state is: {predicted_emotion}")
