# 🌾 Crop Recommendation System

A machine learning-based crop recommendation system that suggests the most suitable crops to grow based on soil nutrients and environmental conditions. This project uses a Random Forest Classifier with 99% accuracy to predict optimal crops from 22 different crop varieties.

## 📋 Project Overview

This system analyzes seven key parameters:
- **Soil Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)
- **Environmental Factors**: Temperature, Humidity, Soil pH
- **Precipitation**: Annual Rainfall

Based on these inputs, the system recommends the most suitable crop for cultivation from a dataset of 22 different crops.

## 🚀 Features

- **High Accuracy**: 99% prediction accuracy using Random Forest Classifier
- **Interactive Web Interface**: User-friendly Gradio-based web application
- **Real-time Predictions**: Instant crop recommendations
- **Comprehensive Crop Database**: Supports 22 different crop varieties
- **Robust Model**: Trained on 2,200 data points with balanced distribution

## 🛠️ Technology Stack

- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Web Interface**: Gradio
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Pickle

## 📊 Supported Crops

The system can recommend the following 22 crops:

### Grains & Cereals
- Rice, Maize

### Legumes & Pulses
- Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil

### Fruits
- Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut

### Commercial Crops
- Cotton, Jute, Coffee

## 📁 Project Structure

```
Crop_prediction/
├── app.py                    # Main application with Gradio interface
├── crop_reco.ipynb          # Jupyter notebook with model development
├── model.pkl                # Trained Random Forest model
├── Crop_recommendation.csv  # Dataset with 2,200 samples
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Crop_prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The web interface will be available at `http://localhost:7860`

## 📊 Model Performance

### Algorithm Comparison
| Model | Accuracy |
|-------|----------|
| Random Forest | **99.32%** |
| XGBoost | 98.64% |
| Gradient Boosting | 98.18% |
| Logistic Regression | 96.36% |
| AdaBoost | 31.59% |

### Final Model Specifications
- **Algorithm**: Random Forest Classifier
- **Parameters**: 
  - n_estimators: 100
  - max_depth: 15
  - random_state: 42
  - n_jobs: -1
- **Training Accuracy**: 100%
- **Testing Accuracy**: 99.32%

## 🎯 How to Use

1. **Launch the web application** by running `python app.py`
2. **Enter soil parameters**:
   - Nitrogen (N): 0-150 ppm
   - Phosphorus (P): 0-150 ppm
   - Potassium (K): 0-210 ppm
3. **Enter environmental conditions**:
   - Temperature: 8-44°C
   - Humidity: 14-100%
   - Soil pH: 3.5-10
   - Rainfall: 20-300 mm
4. **Click "Predict Crop"** to get recommendations
5. **View results** with the recommended crop displayed prominently

## 📈 Dataset Statistics

- **Total Samples**: 2,200
- **Features**: 7 (N, P, K, temperature, humidity, pH, rainfall)
- **Target Classes**: 22 crops
- **Distribution**: Perfectly balanced (100 samples per crop)
- **Data Quality**: No missing values

## 🔬 Model Development Process

1. **Data Preprocessing**:
   - Label encoding for crop names
   - Feature scaling using StandardScaler
   - Train-test split (80:20 ratio)

2. **Feature Engineering**:
   - Correlation analysis
   - Feature importance evaluation
   - Multicollinearity check

3. **Model Selection**:
   - Evaluated 5 different algorithms
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation for robust evaluation

4. **Performance Evaluation**:
   - Accuracy metrics
   - Confusion matrix analysis
   - Classification report generation

## 📝 Input Parameter Guidelines

### Soil Nutrients (NPK)
- **Nitrogen (N)**: Essential for leaf growth, use soil test reports
- **Phosphorus (P)**: Important for root development and flowering
- **Potassium (K)**: Crucial for overall plant health and disease resistance

### Environmental Factors
- **Temperature**: Use average growing season temperature for your region
- **Humidity**: Provide relative humidity during the growing season
- **Soil pH**: Measure using a soil pH meter or laboratory analysis
- **Rainfall**: Use annual precipitation data for your location

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Mahbub Ul Alam Bhuiyan (Nibir)**
- Machine Learning Engineer
- Agricultural Technology Enthusiast

## 🔮 Future Enhancements

- [ ] Integration with weather APIs for real-time data
- [ ] Mobile application development
- [ ] Additional crop varieties
- [ ] Soil health analysis features
- [ ] Yield prediction capabilities
- [ ] Multi-language support

## 📞 Support

For any queries or support, please:
- Open an issue on GitHub
- Contact the author directly
- Check the documentation for common issues

---

**Note**: This system provides recommendations based on historical data and machine learning models. Always consult with local agricultural experts before making important farming decisions.
