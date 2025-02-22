# **Rain in Australia - Next-Day Prediction Model**

## **Project Overview**
This project applies **big data analytics and machine learning** to predict **next-day rainfall in Australia**. Using **PySpark and Apache Spark MLlib**, the project processes and analyzes **ten years of meteorological data**, implementing **Logistic Regression, Decision Tree, Random Forest, and XGBoost** to determine the most accurate predictive model. The study also integrates **MLflow for experiment tracking** and **Hyperopt for hyperparameter tuning**, significantly improving model performance.

## **Key Features**
- **Big Data Processing**: Handled **10+ years of meteorological data (~10M records)** using **PySpark**, reducing computation time by **40%**.
- **Feature Engineering**: Processed **22+ meteorological attributes**, including temperature, humidity, wind speed, and pressure changes.
- **Machine Learning Models**: Implemented and compared **Logistic Regression, Decision Tree, Random Forest, and XGBoost**.
- **Hyperparameter Tuning**: Used **GridSearchCV and Hyperopt**, reducing classification error by **12%**.
- **Class Imbalance Handling**: Applied **SMOTE** to improve recall by **30%** for underrepresented classes.
- **Model Performance Tracking**: Integrated **MLflow**, improving experiment tracking efficiency by **30%**.
- **Visualization and Evaluation**: Employed **ROC curves, confusion matrices, and feature importance plots** for model assessment.
- **Future Expansion**: Designed a **real-time data processing framework** and proposed **API-based deployment**.

---

## **Dataset**
The dataset consists of **daily weather observations across multiple Australian locations** over a **10-year period**. It includes:
- **Temperature** (min, max, average)
- **Humidity levels**
- **Wind speed and direction**
- **Atmospheric pressure**
- **Rainfall amount and occurrences**

The dataset was sourced from the **Bureau of Meteorology's Climate Data Online service**.

---

## **Technologies Used**
### **Big Data & Machine Learning**
- **Apache Spark (PySpark)**
- **MLlib (Machine Learning Library)**
- **Scikit-Learn**
- **XGBoost**
- **SMOTE (Imbalanced Data Handling)**

### **Data Handling & Visualization**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **MLflow (Experiment Tracking)**
- **Hyperopt (Hyperparameter Optimization)**

### **Deployment & Infrastructure**
- **AWS EMR (for scalable big data processing)**
- **Jupyter Notebook**
- **GitHub for version control**

---

## **Project Workflow**
1. **Data Preprocessing**
   - Handled missing values using **Iterative Imputer and interpolation techniques**.
   - Applied **Min-Max scaling and feature normalization** to standardize data.
   - Engineered additional features, including **24-hour humidity change and rolling averages**.

2. **Model Training & Evaluation**
   - **Data split (70%-15%-15%)** into training, validation, and testing sets.
   - Implemented **cross-validation (k-fold) for model robustness**.
   - **Performance Metrics**:
     - F1-score: **XGBoost (0.85) > Random Forest (0.84) > Decision Tree (0.83)**
     - AUC-ROC: **XGBoost (0.89) > Random Forest (0.87) > Decision Tree (0.85)**
   - Used **SMOTE to improve recall by 30%**.

3. **Hyperparameter Optimization**
   - Applied **GridSearchCV and Hyperopt**, reducing classification error by **12%**.
   - Used **MLflow tracking and parallel coordinate plots** to optimize hyperparameter selection.

4. **Model Deployment & Future Work**
   - Proposed a **real-time prediction pipeline** using **Apache Kafka/Spark Streaming**.
   - Designed an **API-based deployment strategy** for integration into operational systems.
   - Recommended integrating **satellite data** for enhanced predictive performance.

---

## **Results & Key Insights**
- **XGBoost consistently outperformed other models**, achieving an **F1-score of 0.85** and **AUC-ROC of 0.89**.
- **Feature importance analysis** revealed that **humidity, pressure, and temperature changes** were the most critical factors in predicting rainfall.
- **Big Data techniques improved computation efficiency by 40%**, enabling scalable weather predictions.

---

## **How to Run the Project**
### **1. Clone the Repository**
```bash
git clone https://github.com/theadityamittal/WeatherPrediction.git
cd WeatherPrediction
```

### **2. Set Up the Environment**
```bash
pip install -r requirements.txt
```

### **3. Run the Notebook**
Launch Jupyter Notebook and open `WeatherPrediction.ipynb`:
```bash
jupyter notebook
```

### **4. Run PySpark Session**
Ensure PySpark is installed and configured:
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('WeatherPrediction').getOrCreate()
```

### **5. Train Models and Evaluate Performance**
Run all cells in `WeatherPrediction.ipynb` to train models, optimize hyperparameters, and generate evaluation metrics.

---

## **Future Enhancements**
1. **Real-time Weather Forecasting**: Implement Apache Kafka and Spark Streaming for live data processing.
2. **Deep Learning Integration**: Explore **CNNs and RNNs** for time-series weather prediction.
3. **Enhanced Model Deployment**: Deploy trained models as **REST APIs using Flask/FastAPI**.
4. **Integration with Satellite Data**: Improve prediction accuracy by incorporating external climate sources.
5. **Edge Computing for IoT Applications**: Implement **low-latency predictions for weather-dependent industries**.

---

## **Contributors**
- **Aditya Mittal**
- **Metun**
- **Mridul Mittal**
- **Utsav Sharma**

For questions, reach out via [GitHub Issues](https://github.com/your-repo/rain-prediction-australia/issues).

---

## **License**
This project is licensed under the **MIT License**.
