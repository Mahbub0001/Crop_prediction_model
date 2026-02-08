import gradio as gr
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Load the Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Build mapping from numeric class -> label string using the dataset's labels
try:
    df_labels = pd.read_csv('Crop_recommendation.csv')
    le = LabelEncoder()
    le.fit(df_labels['label'].astype(str).values)
    INDEX_TO_LABEL = {i: lbl for i, lbl in enumerate(le.classes_)}
except Exception:
    # fallback: a minimal mapping if CSV isn't available
    INDEX_TO_LABEL = {
        0: "rice", 1: "maize", 2: "chickpea", 3: "kidneybeans", 4: "pigeonpeas",
        5: "mothbeans", 6: "mungbean", 7: "blackgram", 8: "lentil", 9: "pomegranate",
        10: "banana", 11: "mango", 12: "grapes", 13: "watermelon", 14: "muskmelon",
        15: "apple", 16: "orange", 17: "papaya", 18: "coconut", 19: "cotton",
        20: "jute", 21: "coffee"
    }

def predict_crop(N,P,K,temperature,humidity,ph,rainfall):
    input_df = pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    # pass a numpy array (no feature names) to avoid sklearn warning
    prediction = model.predict(input_df.values)
    # Ensure we have a scalar integer class label
    pred = int(np.asarray(prediction).ravel()[0])
    crop = INDEX_TO_LABEL.get(pred, str(pred))
    return f"Recommended Crop: {crop}"

inputs = [
    gr.Number(label="Nitrogen (N)"),
    gr.Number(label="Phosphorous (P)"),
    gr.Number(label="Potassium (K)"),
    gr.Number(label="Temperature (°C)"),
    gr.Number(label="Humidity (%)"),
    gr.Number(label="pH"),
    gr.Number(label="Rainfall (mm)")
]

app = gr.Interface(
    fn=predict_crop,
    inputs=inputs,
    outputs="text",
    title="Crop Recommendation System By Nibir"
)
if __name__ == "__main__":
    app.launch(share=False)