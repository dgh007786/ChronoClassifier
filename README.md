# **FX Rate Analysis Pipeline**

## **Overview**
This project implements a real-time **Foreign Exchange (FX) Rate Analysis Pipeline** to fetch, process, and analyze currency pair data using multithreaded Python applications. The pipeline integrates with **Polygon.io** API to retrieve FX rates, stores the processed data in both **SQL** (SQLite) and **MongoDB** databases, computes correlations, and performs predictive modeling using **PyCaret** regression models.

### **Key Features**
- Real-time FX data fetching and processing.
- Multi-database support: SQLite for structured storage and MongoDB for flexible NoSQL storage.
- Multithreaded processing for multiple currency pairs.
- Correlation analysis and predictive modeling for trend analysis.
- Use of **PyCaret** to build and evaluate regression models.
- Scalable infrastructure using Docker for deployment (optional).

## **Setup and Installation**

### **Requirements**
- **Python**: Version 3.8+
- **Databases**:
  - MongoDB (local instance)
  - SQLite (built-in with Python)
- **Polygon.io API Key**

### **Install Required Libraries**
Install the required Python libraries via pip:
```bash 
pip install requests pandas sqlalchemy pymongo pycaret scikit-learn
```

### **Configure API Key**
Set your Polygon.io API key in the API_KEY variable inside the script:
API_KEY = 'your_api_key_here'

### **Database Setup**
SQLite:
The script will automatically create two SQLite database files: fx_data_multiple.db and final_fx_data_multiple.db.
Tables (fx_rates_new_hw2, final_fx_rates_hw2) will be initialized on the first run.
MongoDB:
Ensure a local MongoDB instance is running on localhost:27017.
Collections (fx_rates_a, fx_rates_f) will be created automatically.

### **Predictive Modeling**
Built using PyCaret, which automatically selects the best regression model based on the dataset.
Evaluation is performed using Mean Absolute Error (MAE).

### **Future Enhancements**
Add Docker support for easy deployment.
Integrate cloud services (e.g., AWS RDS, S3) for scalable storage and processing.
Extend support for more currency pairs and historical data.

### **Contributors**
Gunjan Dayani
