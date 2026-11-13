
ML Prediction API

A machine learning prediction API built using Python, Scikit-learn, and FastAPI.

Project Overview

This project trains a Logistic Regression model using the Breast Cancer Wisconsin dataset and provides predictions through a REST API.

Technologies Used

- Python
- Scikit-learn
- NumPy
- FastAPI
- Pydantic
- Joblib

Project Files

- "train.py" – Trains the machine learning model.
- "app.py" – Creates the prediction API.
- "cancer_model.joblib" – Stores the trained model.
- "requirements.txt" – Lists project dependencies.

How It Works

1. Load the dataset.
2. Split data into training and testing sets.
3. Scale features and train a Logistic Regression model.
4. Save the trained model.
5. Load the model into the FastAPI application.
6. Accept input features and return a prediction with confidence.

Run the Project

Install dependencies:

pip install -r requirements.txt

Train and save the model:

python train.py

Start the API:

uvicorn app:app --reload

Open the interactive API documentation at:

"http://127.0.0.1:8000/docs"

API Endpoint

"POST /predict"

Accepts 30 numerical features and returns a predicted class and confidence score.

Note

This is an educational machine learning project. It is not intended for medical diagnosis or clinical decision-making.
