# Cholesterol_Prediction
Cholesterol prediction

Project link : https://cholesterol-prediction.herokuapp.com/
# 🩺 Cholesterol Level Prediction

A **Machine Learning project** to predict cholesterol levels based on **animal BMI and weight**.  
The model was developed using **Python, Scikit-learn, Pandas, NumPy, and Flask**, with deployment on cloud platforms for real-time predictions.  

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Results](#-results)
- [Deployment](#-deployment)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [References](#-references)

---

## 📖 Overview
The objective of this project is to build a predictive model that estimates **cholesterol levels** based on animal characteristics like BMI and weight.  
This can be applied in healthcare research for identifying risk patterns and supporting preventive care strategies.

---

## 📊 Dataset
- **Source:** Custom dataset (BMI & weight vs cholesterol levels)  
- **Features:**  
  - Body Mass Index (BMI)  
  - Weight (kg)  
  - Cholesterol Level (target variable)  

---

## ⚙️ Approach
1. **Data Preprocessing**  
   - Handling missing values and outliers  
   - Normalization & scaling  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized distributions and correlations  
   - Identified trends between BMI, weight, and cholesterol  

3. **Model Development**  
   - Algorithms used: **Linear Regression, Ridge, Lasso, Random Forest, XGBoost**  
   - Hyperparameter tuning with **GridSearchCV**  
   - Cross-validation for performance stability  

4. **Evaluation**  
   - Metrics: **R² Score, RMSE**  
   - Residual/error analysis  

---

## 🛠 Tech Stack
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, XGBoost  
- **Deployment**: Flask, Heroku  
- **Tools**: Jupyter Notebook, Google Colab  

---

## 🚀 Features
- Input BMI and weight → Predict cholesterol level  
- Interactive web-based interface (Flask/Streamlit)  
- Multiple ML models compared and benchmarked  
- Deployed for **real-time access**  

---

## 📈 Results
- **Best Model:** XGBoost  
- **Performance:**  
  - R² Score: *0.89*  
  - RMSE: *< value >*  
- Validated model stability with residual plots  

*(Add plots/screenshots of EDA and prediction outputs here)*  

---

## 🌐 Deployment
🔗 Live Demo: [Cholesterol Prediction App](https://cholesterol-prediction.herokuapp.com/)  

---

## 🖥 How to Run

```bash
# Clone repository
git clone https://github.com/yourusername/Cholesterol_Prediction.git

# Navigate to project folder
cd Cholesterol_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py


🔮 Future Improvements

- Collect larger real-world healthcare datasets
- Add more features (age, diet, activity level)
- Deploy using Docker + CI/CD pipelines
- Build API for integration with healthcare platforms
- Explore deep learning models for prediction

📚 References
Scikit-learn Documentation : https://scikit-learn.org/stable/
XGBoost Documentation: https://xgboost.readthedocs.io/en/stable/
Healthcare Data Analysis Examples: https://www.kaggle.com/

Developed by Manas Kumar Bej
