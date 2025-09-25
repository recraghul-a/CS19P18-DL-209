import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. Load the Iris dataset (no need to download, it's included with scikit-learn)
from sklearn.datasets import load_iris
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Convert labels to one-hot encoding
y = to_categorical(y)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize the data (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the Sequential Neural Network model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu')) # Input layer with 4 features
model.add(Dense(8, activation='relu')) # Hidden layer
model.add(Dense(3, activation='softmax')) # Output layer with 3 classes

# 5. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model's structure
model.summary()

# 6. Train the model
print("\nTraining the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1, validation_data=(X_test, y_test))

# 7. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")