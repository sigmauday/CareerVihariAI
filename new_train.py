import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the new intents file with error handling
try:
    with open('intent_new.json', 'r') as file:
        data = file.read()
        if not data.strip():  # Check if file is empty
            raise ValueError("intent_new.json is empty!")
        # First attempt to parse as JSON
        intents = json.loads(data)
    print("JSON file loaded successfully!")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print("Please check the syntax of intent_new.json.")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Check the type of intents and handle different structures
print("Type of intents:", type(intents))
if isinstance(intents, str):
    # If intents is a string, parse it again
    try:
        intents = json.loads(intents)
        print("Parsed string into JSON object.")
    except json.JSONDecodeError as e:
        print(f"Error parsing string as JSON: {e}")
        print("intent_new.json contains a string that is not valid JSON.")
        exit(1)
elif isinstance(intents, dict):
    # If intents is a dictionary, check for a nested 'intents' key
    if 'intents' in intents:
        intents = intents['intents']
        print("Found nested 'intents' key, using that list.")
    else:
        print("intent_new.json is a dictionary but does not contain an 'intents' key.")
        exit(1)
elif not isinstance(intents, list):
    print(f"Unexpected type for intents: {type(intents)}. Expected a list.")
    exit(1)

# Verify that intents is a list of dictionaries
if not intents:
    print("intents list is empty!")
    exit(1)

# Initialize lists for words, classes, and training data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process each intent
for intent in intents:
    # Check the type of intent
    if not isinstance(intent, dict):
        print(f"Error: Expected intent to be a dictionary, but got {type(intent)}: {intent}")
        exit(1)
    # Check if 'patterns' key exists
    if 'patterns' not in intent:
        print(f"Error: Intent is missing 'patterns' key: {intent}")
        exit(1)
    # Check if 'tag' key exists
    if 'tag' not in intent:
        print(f"Error: Intent is missing 'tag' key: {intent}")
        exit(1)
    for pattern in intent['patterns']:
        # Tokenize each pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents (pattern, intent tag)
        documents.append((word_list, intent['tag']))
        # Add intent tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes to new .pkl files
with open('words_new.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes_new.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Create training data
training = []
output_empty = [0] * len(classes)

# Vectorize patterns using TF-IDF
patterns = [' '.join([lemmatizer.lemmatize(word.lower()) for word in doc]) for doc, _ in documents]
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(patterns).toarray()

# Save the vectorizer
with open('vectorizer_new.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Create labels (one-hot encoded)
y = []
for doc, tag in documents:
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    y.append(output_row)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Shuffle the data
combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = np.array(X)
y = np.array(y)

# Build a simple neural network model
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model_new.h5')

print("Training completed! New .pkl files and model saved as: words_new.pkl, classes_new.pkl, vectorizer_new.pkl, and model_new.h5")