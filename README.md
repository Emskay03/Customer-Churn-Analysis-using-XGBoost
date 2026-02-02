# Customer Churn Prediction System

An end-to-end machine learning system for predicting customer churn using XGBoost and FastAPI.  
The project focuses on handling severe class imbalance, optimizing business-relevant metrics, and exposing predictions through a REST API.

---

## 📌 Project Overview

Customer churn is a critical problem in subscription-based businesses.  
This project builds a predictive system to identify high-risk customers using historical usage and behavior data.

The system includes:

- Data preprocessing and analysis
- Imbalanced classification using XGBoost
- Threshold optimization for business cost
- REST API for real-time inference
- Model serialization and schema validation

---

## 🚀 Features

- Handles extreme class imbalance using class weighting
- Optimizes recall and ROC-AUC for churn detection
- Uses XGBoost for high-performance classification
- Provides RESTful prediction endpoint using FastAPI
- Input validation using Pydantic
- Tested using Swagger UI and Postman
- Reproducible training and inference pipeline

---

## 🛠️ Tech Stack

- Programming Language: Python  
- Machine Learning: XGBoost, Scikit-learn  
- API Framework: FastAPI  
- Data Processing: Pandas, NumPy  
- Model Serialization: Joblib  
- Validation: Pydantic  
- Testing: Postman, Swagger UI  

---

## 📊 Dataset

- Telecom customer churn dataset (50K+ records)
- Highly imbalanced (~47:1 non-churn to churn ratio)
- Preprocessed and cleaned before training

Target variable: `churn` (0 = No Churn, 1 = Churn)

---

## ⚙️ Project Structure

churn-project/
│

├── app/

│ └── main.py # FastAPI application

│

├── data/

│ └── clean_churn.csv # Preprocessed dataset

│

├── model/

│ ├── churn_xgb.pkl # Trained model

│ └── columns.json # Feature schema

│

├── train.py # Model training script

├── requirements.txt # Dependencies

└── README.md

---

## 🔍 Methodology

### 1. Data Preparation
- Removed duplicates and invalid entries
- Checked data types and distributions
- Analyzed class imbalance
- Performed basic feature engineering

### 2. Model Training
- Used XGBoost classifier
- Applied `scale_pos_weight` to handle imbalance
- Tuned hyperparameters for stability
- Used stratified train-test split

### 3. Evaluation
- Confusion matrix
- Precision, recall, F1-score
- ROC-AUC score
- Threshold tuning for cost optimization

### 4. Inference Pipeline
- Serialized model using Joblib
- Stored feature schema in JSON
- Built FastAPI service
- Added Pydantic input validation
- Tested endpoints via Postman and Swagger UI

---

## 📈 Results

- ROC-AUC: ~0.97  
- Churn Recall: ~90%  
- Balanced tradeoff between false positives and missed churn cases

The model prioritizes identifying potential churners while managing operational costs.

---

## ▶️ How to Run Locally

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Train the Model
python train.py

### 3. Start the API Server
python -m uvicorn app.main:app --reload
Server will start at: http://127.0.0.1:8000

### 4. Access API Documentation
Open in browser: http://127.0.0.1:8000/docs
Use the Swagger UI to test predictions.


### 📡 API Usage
Endpoint
POST /predict

Example Request

     {"tenure_months": 12,
   
     "monthly_usage_hours": 40,
   
     "has_multiple_devices": 1,
   
     "customer_support_calls": 2,
   
     "payment_failures": 1,
   
     "is_premium_plan": 0}

Example Response

    {"churn_probability": 0.74,
  
    "churn_prediction": 1}

