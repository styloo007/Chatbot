from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# Load intents from JSON file
with open('C:\\Users\\shash\\Downloads\\ChatBot\\intents.json') as file:
    intents = json.load(file)

# Extract patterns and tags
patterns = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
        
        
        
# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)

# Padding
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)



# Convert tags to one-hot encoding
num_classes = len(set(tags))
tag_indices = {tag: i for i, tag in enumerate(set(tags))}
one_hot_tags = np.zeros((len(tags), num_classes))
for i, tag in enumerate(tags):
    one_hot_tags[i, tag_indices[tag]] = 1


# Define Model
embedding_dim = 16
model = Sequential([
    Embedding(len(word_index) + 1, embedding_dim, input_length=max_sequence_length),
    LSTM(64),
    Dense(num_classes, activation='softmax')
])

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train Model
model.fit(padded_sequences, one_hot_tags, epochs=249, verbose=1)


# Define a function to get a response from the model
def get_response(model, tokenizer, max_sequence_length, tag_indices, input_text):
    # Tokenize and pad the input text
    test_sequence = tokenizer.texts_to_sequences([input_text])
    padded_test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length)

    # Predict the tag for the input text
    predictions = model.predict(padded_test_sequence)
    predicted_tag_index = np.argmax(predictions)
    predicted_tag = [tag for tag, index in tag_indices.items() if index == predicted_tag_index][0]

    # Find the response for the predicted tag
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            responses = intent['responses']
            return np.random.choice(responses)



def index(request):
    if request.method == 'POST':
        # Get user input from the form
        user_input = request.POST.get('user_input', '')
        print(user_input)

        # Generate response using the chatbot model
        response = get_response(model, tokenizer, max_sequence_length, tag_indices, user_input)
        print(response)

        # Prepare response data
        data = {
            'user_input': user_input,
            'response': response
        }

        # Print the user input
        print("User Input:", user_input)

        # Return JSON response with the chatbot's response
        return JsonResponse(data)

    # Render the chatbot HTML template for GET requests
    return render(request, 'index.html')
