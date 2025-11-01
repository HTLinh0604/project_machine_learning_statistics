# 🤖 Machine Learning Project Collection  
*(Tổng Hợp Các Dự Án Machine Learning Trong Môn Học "Máy Học Thống Kê")*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-ff6f00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-red?logo=keras)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-013243?logo=plotly)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-5C9EAD)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Statistical%20Models-6E7E85)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv)
![PMDARIMA](https://img.shields.io/badge/pmdarima-Time%20Series-003B73)

---

## 👋 Welcome / Chào mừng  
Welcome to my course project repository for *Statistical Machine Learning*.  
*(Chào mừng bạn đến với kho lưu trữ các dự án trong đồ án môn học “Máy Học Thống Kê”.)*  

This repository contains several projects demonstrating my skills in **data processing, model building, and evaluation**.  
*(Kho này chứa nhiều dự án thể hiện kỹ năng của tôi trong xử lý dữ liệu, xây dựng và đánh giá mô hình học máy.)*

---

## 📋 Table of Contents *(Mục Lục)*  
1. [💳 Project 1: Credit Score Classification *(Phân loại điểm tín dụng)*](#-project-1-credit-score-classification)  
2. [👥 Project 2: Bank Customer Clustering *(Phân cụm khách hàng ngân hàng)*](#-project-2-bank-customer-clustering)  
3. [💵 Project 3: Fake Currency Detection *(Phân loại tiền thật/giả)*](#-project-3-fake-currency-detection)  
4. [💎 Project 4: Diamond Price Prediction *(Dự đoán giá kim cương)*](#-project-4-diamond-price-prediction)  
5. [📈 Project 5: Uber Stock Price Forecast *(Dự đoán giá cổ phiếu Uber)*](#-project-5-uber-stock-price-forecast)  
6. [🍊 Project 6: Orange Quality Classification Using CNN *(Phân loại chất lượng cam bằng CNN)*](#-project-6-orange-quality-classification-using-cnn)  

---

## 💳 Project 1: Credit Score Classification  
*(Phân loại điểm tín dụng)*  

### 🎯 Overview *(Giới thiệu)*  
This project builds a model to predict customers’ credit score (Poor, Standard, Good) using financial and personal data.  
*(Dự án này tập trung xây dựng mô hình dự đoán điểm tín dụng của khách hàng dựa trên thông tin cá nhân và lịch sử tài chính.)*

### ⚙️ Workflow *(Quy trình thực hiện)*  
1️⃣ **Data Loading & Exploration / Tải và khám phá dữ liệu**  
2️⃣ **Cleaning & Preprocessing / Làm sạch và tiền xử lý dữ liệu**  
3️⃣ **Feature Engineering / Kỹ thuật đặc trưng**  
4️⃣ **Feature Selection & Scaling / Lựa chọn & chuẩn hóa đặc trưng**  
5️⃣ **Model Training / Huấn luyện mô hình** (`KNN`, `SVM`)  
6️⃣ **Evaluation / Đánh giá mô hình**

### 🏆 Results *(Kết quả)*  
> - **KNN Accuracy:** 71% (Best)  
> - **SVM Accuracy:** 51%  

### 💻 Tech Stack *(Công nghệ sử dụng)*  
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`

---

## 👥 Project 2: Bank Customer Clustering  
*(Phân cụm khách hàng ngân hàng)*  

### 🎯 Overview  
This unsupervised learning project clusters bank customers using their demographic and transaction data.  
*(Dự án sử dụng kỹ thuật học không giám sát để phân nhóm khách hàng ngân hàng dựa trên hành vi và nhân khẩu học.)*

### ⚙️ Workflow  
- **EDA & Cleaning / Khám phá & làm sạch dữ liệu**  
- **RFM Analysis / Phân tích RFM (Recency, Frequency, Monetary)**  
- **Visualization / Trực quan hóa dữ liệu**  
- **Modeling / Xây dựng mô hình phân cụm (KMeans, DBSCAN)**  
- **PCA Visualization / Giảm chiều & trực quan hóa**

### 🏆 Results  
> - **K-Means + PCA:** Silhouette Score = 0.2956  
> - **DBSCAN:** Silhouette = 0.3608 (nhưng 39.9% dữ liệu nhiễu)

### 💻 Tech Stack  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`, `kneed`

---

## 💵 Project 3: Fake Currency Detection  
*(Phân loại tiền thật/giả)*  

### 🎯 Overview  
Binary classification model distinguishing genuine vs fake banknotes using physical measurements.  
*(Mô hình phân biệt tiền thật/giả dựa trên các đặc trưng vật lý như chiều dài, chiều cao, lề.)*

### 🏆 Results  
> - **KNN:** Accuracy 99%  
> - **Logistic Regression:** Accuracy 98%

### 💻 Tech Stack  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`

---

## 💎 Project 4: Diamond Price Prediction  
*(Dự đoán giá kim cương)*  

### 🎯 Overview  
Regression project predicting diamond price using features such as carat, cut, color, clarity, etc.  
*(Dự án hồi quy dự đoán giá kim cương dựa trên các thuộc tính vật lý và phân loại.)*

### 🏆 Results  
> - **Random Forest:** R² = 0.9784, MAE = 0.0151  
> - **Decision Tree:** R² = 0.9616  
> - **Linear Regression:** R² = 0.8129  

### 💻 Tech Stack  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`

---

## 📈 Project 5: Uber Stock Price Forecast  
*(Dự đoán giá cổ phiếu Uber)*  

### 🎯 Overview  
Forecasting project using ARIMA to predict Uber stock prices (2019–2025).  
*(Dự án phân tích chuỗi thời gian dự đoán giá cổ phiếu Uber bằng ARIMA.)*

### 🏆 Results  
> - **Best Model:** ARIMA(2,1,2)  
> - **Test RMSE:** 0.115  

### 💻 Tech Stack  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `pmdarima`

---

## 🍊 Project 6: Orange Quality Classification Using CNN  
*(Phân loại chất lượng cam bằng CNN)*  

### 🎯 Overview  
CNN model classifying oranges as *Good* or *Bad* based on image data.  
*(Mô hình CNN phân loại hình ảnh cam chất lượng tốt/xấu.)*

### 🏆 Results  
> - **Accuracy:** 72.5% (Test set)  
> - **Overfitting observed**  

### 💻 Tech Stack  
`pandas`, `numpy`, `matplotlib`, `opencv-python`, `tensorflow`, `keras`, `scikit-learn`

