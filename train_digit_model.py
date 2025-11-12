from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import streamlit as st

st.title("Hello, Streamlit!")
st.write("If this works, the problem is in your app code.")


# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    to_categorical(y_train),
    epochs=5,
    validation_data=(x_test, to_categorical(y_test))
)

# Save the trained model
model.save("digit_model.h5")
print("✅ Model saved as digit_model.h5")

