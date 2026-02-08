import pandas as pd
import numpy as np
import pickle
import gradio as gr
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Crop mapping dictionary
details = {
    'rice': 0,
    'maize': 1,
    'chickpea': 2,
    'kidneybeans': 3,
    'pigeonpeas': 4,
    'mothbeans': 5,
    'mungbean': 6,
    'blackgram': 7,
    'lentil': 8,
    'pomegranate': 9,
    'banana': 10,
    'mango': 11,
    'grapes': 12,
    'watermelon': 13,
    'muskmelon': 14,
    'apple': 15,
    'orange': 16,
    'papaya': 17,
    'coconut': 18,
    'cotton': 19,
    'jute': 20,
    'coffee': 21
}

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct file paths
csv_path = os.path.join(script_dir, "Crop_recommendation.csv")
model_path = os.path.join(script_dir, "model.pkl")
scaler_path = os.path.join(script_dir, "scaler.pkl")

# Initialize variables
final_model = None
scaler = None

# Load the dataset for preprocessing and fit scaler
print("Loading dataset for scaler fitting...")
try:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Create label encoding
        unique_labels = df['label'].unique()
        label_map = {label: i for i, label in enumerate(unique_labels)}
        df['label_encoded'] = df['label'].map(label_map)
        
        # Prepare features
        df_2 = df.drop(['label'], axis=1)
        X = df_2.drop('label_encoded', axis=1)
        
        # Fit scaler on the entire dataset (same as training approach)
        scaler = StandardScaler()
        scaler.fit(X)
        print("Scaler fitted from dataset successfully!")
    else:
        print(f"Warning: CSV file not found at {csv_path}")
        print("Attempting to load pre-saved scaler...")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded from pickle file!")
        else:
            print("Warning: Both CSV and scaler pickle not found. Creating default scaler...")
            scaler = StandardScaler()
            # Fit with some sample ranges (fallback)
            sample_data = np.array([[25, 25, 25, 25, 70, 6.5, 100]])
            scaler.fit(sample_data)
except Exception as e:
    print(f"Error loading dataset: {e}")
    scaler = StandardScaler()

# Load the pre-trained model
print("Loading pre-trained model...")
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            final_model = pickle.load(file)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file not found at {model_path}")
        raise FileNotFoundError(f"model.pkl not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

print("All components loaded successfully!")

# Create reverse mapping for predictions
reverse_details = {v: k for k, v in details.items()}

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """
    Predict the best crop based on soil and weather parameters
    
    Args:
        nitrogen (float): N value in soil
        phosphorus (float): P value in soil
        potassium (float): K value in soil
        temperature (float): Temperature in Celsius
        humidity (float): Humidity percentage
        ph (float): pH value of soil
        rainfall (float): Rainfall in mm
    
    Returns:
        str: Predicted crop name
    """
    try:
        # Check if model and scaler are loaded
        if final_model is None:
            return "❌ Error: Model not loaded. Please check if model.pkl exists."
        if scaler is None:
            return "❌ Error: Scaler not loaded. Please check your data files."
        
        # Create DataFrame from input values
        input_data = pd.DataFrame(
            [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )
        
        # Scale the input data using the fitted scaler
        scaled_input_data = scaler.transform(input_data)
        
        # Make prediction using the trained model
        predicted_label_encoded = final_model.predict(scaled_input_data)
        
        # Map the predicted encoded label back to the original crop name
        predicted_crop_name = reverse_details[predicted_label_encoded[0]]
        
        return f"✅ Recommended Crop: **{predicted_crop_name.upper()}**"
    
    except Exception as e:
        return f"❌ Error in prediction: {str(e)}"

# Create Gradio Interface
def create_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(title="Crop Recommendation System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌾 Crop Recommendation System")
        gr.Markdown("Enter soil and weather parameters to get the best crop recommendation for your farm.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Soil Nutrients (NPK)")
                nitrogen = gr.Number(
                    label="Nitrogen (N)",
                    value=50,
                    info="Nitrogen content in soil (0-150)"
                )
                phosphorus = gr.Number(
                    label="Phosphorus (P)",
                    value=50,
                    info="Phosphorus content in soil (0-150)"
                )
                potassium = gr.Number(
                    label="Potassium (K)",
                    value=50,
                    info="Potassium content in soil (0-210)"
                )
            
            with gr.Column():
                gr.Markdown("### Environmental Factors")
                temperature = gr.Number(
                    label="Temperature (°C)",
                    value=25,
                    info="Temperature in Celsius (8-44)"
                )
                humidity = gr.Number(
                    label="Humidity (%)",
                    value=70,
                    info="Humidity percentage (14-100)"
                )
                ph = gr.Number(
                    label="Soil pH",
                    value=6.5,
                    info="pH value of soil (3.5-10)"
                )
            
            with gr.Column():
                gr.Markdown("### Precipitation")
                rainfall = gr.Number(
                    label="Rainfall (mm)",
                    value=100,
                    info="Annual rainfall in mm (20-300)"
                )
        
        with gr.Row():
            predict_btn = gr.Button("🌱 Predict Crop", size="lg", variant="primary")
            clear_btn = gr.Button("Clear", size="lg")
        
        output = gr.Textbox(
            label="Recommended Crop",
            interactive=False,
            text_align="center"
        )
        
        # Connect button to prediction function
        predict_btn.click(
            fn=predict_crop,
            inputs=[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall],
            outputs=output
        )
        
        # Clear button functionality
        clear_btn.click(
            fn=lambda: ("", 50, 50, 50, 25, 70, 6.5, 100),
            outputs=[output, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        )
        
        gr.Markdown("""
        ### 📊 About This Model
        - **Created By**: Mahbub Ul Alam Bhuiyan (Nibir)
        - **Model Type**: Random Forest Classifier
        - **Accuracy**: ~99% on test data
        - **Features**: 7 input parameters (3 soil nutrients, 3 environmental factors, 1 precipitation)
        - **Crops Supported**: 22 different crops (rice, maize, chickpea, kidney beans, pigeon peas, moth beans, mung bean, black gram, lentil, pomegranate, banana, mango, grapes, watermelon, musk melon, apple, orange, papaya, coconut, cotton, jute, coffee)
                    
        ### 💡 Tips for Best Results
        - Use actual soil test reports for NPK values
        - Provide location-specific average temperature and humidity
        - Use past year's rainfall data for your region
        """)
    
    return demo

interface = create_interface()
    # Hugging Face Spaces configuration
interface.launch(
        share=False,
    )
