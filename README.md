# 🧬 Machine Learning Model for Predicting Sertraline-like Activities

A Streamlit web application that predicts sertraline-like antidepressant activity and cancer chemosensitization potential using multiple machine learning algorithms.

## Features

- **31 molecular descriptors** extracted from SMILES strings using RDKit
- **5 ML algorithms**: XGBoost, Random Forest, SVM, Logistic Regression, KNN
- **Bagging SVM** ensemble method (17 estimators)
- **Cross-validation** for robust model evaluation
- **Interactive predictions** for single compounds or batch processing
- **Comprehensive visualizations**: ROC curves, confusion matrices, feature importance

## Dataset

The model uses 268 compounds with activity labels (active/inactive) and SMILES notation.

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run project.py
```

## Deployment on Render

This app is configured for deployment on Render. The following files are included:

- `requirements.txt` - Python dependencies
- `setup.sh` - Configuration script for Streamlit on Render

### Render Deployment Steps:

1. **Create a new Web Service on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure the service:**
   - **Name**: Your app name (e.g., "sertraline-predictor")
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && chmod +x setup.sh && ./setup.sh`
   - **Start Command**: `streamlit run project.py --server.port=$PORT --server.address=0.0.0.0`
   - **Plan**: Free tier is sufficient for testing

3. **Environment Variables** (if needed):
   - No environment variables required for basic deployment

4. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## Usage

1. **Load Dataset**: The app automatically loads `bixuchenggong11.CSV` from the project directory
2. **Train Models**: Click "🚀 Train All Models" to extract features and train all ML models
3. **View Results**: Explore model performance metrics, ROC curves, and confusion matrices
4. **Make Predictions**: Use the prediction interface to predict activity for new compounds

## Model Performance

The app trains and compares multiple ML algorithms:
- XGBoost
- Random Forest
- SVM (with Bagging ensemble)
- Logistic Regression
- K-Nearest Neighbors

Each model is evaluated using:
- Accuracy, Precision, Recall, F1-Score
- 5-fold cross-validation
- ROC curve analysis
- Confusion matrix

## File Structure

```
.
├── project.py              # Main Streamlit application
├── bixuchenggong11.CSV     # Dataset file
├── requirements.txt        # Python dependencies
├── setup.sh               # Render deployment script
└── README.md              # This file
```

## Notes

- The CSV file (`bixuchenggong11.CSV`) must be in the same directory as `project.py`
- Ensure the CSV has columns: `num`, `name`, `activity`, `smiles`
- SMILES strings are automatically cleaned (whitespace removed) during processing

## License

[Add your license here]

## Author

[Add your name/contact here]
