
import tensorflow as tf
import os

try:
    model_path = "models/low_dropout.keras"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
    else:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        model.summary()
        print("Input Shape:", model.input_shape)
        print("Output Shape:", model.output_shape)
except Exception as e:
    print(f"Error inspecting model: {e}")
