import streamlit as st
import numpy as np
from tensorflow import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("digit_model.h5")

st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9) to see the prediction.")

uploaded_file = st.file_uploader("Upload a digit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = ImageOps.invert(image)  # MNIST digits are white on black
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"✅ Predicted Digit: {predicted_digit}")
    st.bar_chart(prediction[0])

