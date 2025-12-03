📈 GDP Prediction Using Machine Learning

A Machine Learning project to predict the Gross Domestic Product (GDP) of countries using various socio-economic indicators.

⭐ Project Overview

This project builds and evaluates machine learning models to predict a country’s GDP based on features such as:

Population

Literacy rate

Inflation

Employment rate

Import & export values

Health & education spending

Internet penetration

And more (depending on dataset)

The goal is to understand how different factors influence GDP and to build a model with good predictive accuracy.

📂 Project Structure
GDP-ML-Project/
│
├── data/
│   ├── raw_dataset.csv
│   └── cleaned_dataset.csv
│
├── notebooks/
│   └── gdp_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── models/
│   └── best_model.pkl
│
├── README.md
└── requirements.txt

🧠 Machine Learning Models Used

The project compares multiple models:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Decision Tree Regressor

Support Vector Regressor (SVR)

Model performance is evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/gdp-ml-project
cd gdp-ml-project

2️⃣ Install Dependencies
pip install -r requirements.txt

🧹 Data Preprocessing

The preprocessing pipeline includes:

Handling missing values

Removing outliers

Encoding categorical variables

Feature scaling (Standard Scaler / MinMax Scaler)

Splitting dataset into train/test sets

🚀 How to Run the Project
▶ Train the Model
python src/train_model.py

▶ Evaluate the Model
python src/evaluate_model.py

📊 Results & Insights

Identifies top factors contributing to GDP

Shows correlation between socio-economic factors and GDP

Provides predictions with trained ML models

Random Forest / XGBoost usually give best performance

(Include your actual metrics here when available)

🔍 Visualization

The project includes:

Correlation heatmaps

Feature importance graphs

GDP prediction vs actual plots

Distribution analysis of variables

📝 Requirements

Example dependencies:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter

🧪 Future Improvements

Add deep learning model (ANN)

Include time-series GDP forecasting

Deploy model using Flask / FastAPI

Build dashboard using Streamlit

🤝 Contributors

swetha priya
Nandini 
Mokshith
Kethan
Uday

(Add team members if any)

📜 License

This project is licensed under the MIT License (or specify your license).
