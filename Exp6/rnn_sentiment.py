import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence

# 1. Load the IMDB dataset
# We'll use the top 10,000 most frequently occurring words.
max_features = 10000
maxlen = 500  # Cut reviews after this number of words (for padding)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# 2. Pad sequences to ensure all reviews have the same length
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# 3. Build the RNN model
model = Sequential()
model.add(Embedding(max_features, 32))  # Embedding layer to convert words to vectors
model.add(SimpleRNN(32))  # Simple RNN layer with 32 units
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Display the model's architecture
model.summary()

# 4. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
print("\nTraining the model...")
model.fit(x_train, y_train,
          epochs=5,
          batch_size=128,
          validation_data=(x_test, y_test))

# 6. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")