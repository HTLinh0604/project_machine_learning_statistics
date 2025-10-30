# 🤖 Tổng Hợp Các Dự Án Machine Learning Trong Môn Học "Máy Học Thống Kê"

👋 Chào mừng bạn đến với kho lưu trữ các dự án trong đồ án môn học 'Máy học thống kê". Kho này chứa một loạt các dự án thể hiện những kỹ năng và kiến thức của tôi trong lĩnh vực học máy, từ xử lý dữ liệu đến xây dựng và đánh giá các mô hình phức tạp.

## 📋 Mục Lục
1. [💳 Dự án 1: Phân loại điểm tín dụng](#-dự-án-1-phân-loại-điểm-tín-dụng)
2. [👥 Dự án 2: Phân cụm khách hàng ngân hàng](#-dự-án-2-phân-cụm-khách-hàng-ngân-hàng)
3. [💵 Dự án 3: Phân loại tiền thật/giả](#-dự-án-3-phân-loại-tiền-thậtgiả)
4. [💎 Dự án 4: Dự đoán giá kim cương](#-dự-án-4-dự-đoán-giá-kim-cương)
5. [📈 Dự án 5: Dự đoán giá cổ phiếu Uber](#-dự-án-5-dự-đoán-giá-cổ-phiếu-uber)
6. [🍊 Dự án 6: Phân loại chất lượng cam bằng CNN](#-dự-án-6-phân-loại-chất-lượng-cam-bằng-cnn)

---

## 💳 Dự án 1: Phân loại điểm tín dụng

### 🎯 Giới thiệu
Dự án này tập trung vào việc xây dựng một mô hình phân loại để dự đoán điểm tín dụng cá nhân thành ba nhóm (**Poor, Standard, Good**) (bài toán Phân loại đa lớp) dựa trên các thông tin cá nhân và lịch sử tài chính.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**: Tải tập dữ liệu train và test, sau đó **gộp lại để thực hiện tiền xử lý đồng bộ**.  
2️⃣ **Làm sạch và tiền xử lý dữ liệu**:  
- Chuẩn hóa văn bản ở cột `Type_of_Loan` (xóa ký tự đặc biệt, khoảng trắng thừa, thay thế chuỗi không hợp lệ bằng `NaN`).  
- Chuyển đổi kiểu dữ liệu từ object sang số hoặc thời gian cho phù hợp.  
- Xử lý các giá trị ngoại lệ (outlier) ở các biến số bằng phương pháp **IQR** (thay thế bằng giá trị trung bình).  
- Xử lý giá trị thiếu (`NaN`) bằng **mode** cho các biến dạng object và bằng **trung bình** cho các biến dạng số.
- Chuẩn hóa dữ liệu bằng **MinMaxScaler**.
  
3️⃣ **Xây dựng và huấn luyện mô hình**:  
- Huấn luyện các mô hình **K-Nearest Neighbors (KNN)** và **Support Vector Machine (SVM)**.
  
4️⃣ **Đánh giá mô hình**:  
- Đánh giá hiệu suất của các mô hình trên tập kiểm tra.  

### 🏆 Kết quả
| Mô hình | Độ chính xác (Accuracy) | F1-Score (Poor) | F1-Score (Good) | Nhận xét |
| :--- | :--- | :--- | :--- | :--- |
| **KNN** | **71%** | 0.61 | 0.75 | Khả năng phân loại **đồng đều và ổn định** hơn. |
| SVM | 51% | 0.03 | - | Hoạt động kém hiệu quả, đặc biệt không nhận diện được nhóm *Poor*. |

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`KNeighborsClassifier`, `SVC`, `MinMaxScaler`)

---

## 👥 Dự án 2: Phân cụm khách hàng ngân hàng

### 🎯 Giới thiệu
Dự án này sử dụng các kỹ thuật phân cụm để phân nhóm khách hàng của một ngân hàng dựa trên hành vi giao dịch và thông tin nhân khẩu học. Mục tiêu là để xác định các phân khúc khách hàng khác nhau nhằm phục vụ cho các chiến lược kinh doanh.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu (EDA)**: Tải và kiểm tra thông tin, dữ liệu thiếu, dữ liệu trùng lặp.  
2️⃣ **Làm sạch dữ liệu**:
- Tính **tuổi** (Age) khách hàng từ ngày sinh, loại bỏ các giá trị không hợp lệ.
- Xử lý các giá trị không hợp lệ trong cột giới tính.
  
3️⃣ **Phân tích RFM (Recency, Frequency, Monetary)**:
- Tính toán các giá trị Recency, Frequency, và Monetary cho mỗi khách hàng.
  
4️⃣ **Tiền xử lý nâng cao**:
- Xử lý outlier bằng phương pháp **IQR** và điền giá trị thiếu (median).
- Chuẩn hóa dữ liệu bằng **StandardScaler**.
  
5️⃣ **Xây dựng mô hình phân cụm**:
- Áp dụng thuật toán **K-Means** (k=5) kết hợp **PCA** (giảm chiều xuống 3 thành phần) và **DBSCAN**.
- Sử dụng phương pháp Elbow và Silhouette để xác định số cụm tối ưu cho K-Means.
  
6️⃣ **Đánh giá và phân tích cụm**:
- Đánh giá mô hình bằng chỉ số Silhouette.
- Trực quan hóa các cụm và phân tích đặc điểm của từng cụm.

### 🏆 Kết quả
| Mô hình | Số cụm (k) | Silhouette Score | Nhận xét |
| :--- | :--- | :--- | :--- |
| **K-Means + PCA** | 5 | **0.2956** | Đạt điểm cao hơn so với K-Means gốc (0.2544). **Được đề xuất tối ưu nhất** cho triển khai. |
| DBSCAN | 8 | 0.3608 | Phù hợp để phát hiện **khách hàng có hành vi bất thường** (39.9% điểm dữ liệu là nhiễu). |

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`, `plotly`
*   `scikit-learn` (`KMeans`, `DBSCAN`, `StandardScaler`, `PCA`)

---

## 💵 Dự án 3: Phân loại tiền thật/giả

### 🎯 Giới thiệu
Mục tiêu của dự án này là xây dựng một mô hình có khả năng phân biệt giữa tiền thật và tiền giả (bài toán Phân loại nhị phân) dựa trên các đặc trưng đo lường vật lý (chiều dài, chiều cao, lề).

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**: Tải dữ liệu và xử lý 37 giá trị thiếu ở cột `margin_low` bằng cách điền bằng giá trị **mode**.  
2️⃣ **Tiền xử lý dữ liệu**:
- **Cân bằng dữ liệu** gốc bị mất cân bằng (1000 mẫu thật / 500 mẫu giả) bằng kỹ thuật **RandomOverSampler**.
- Chuẩn hóa các đặc trưng bằng **MinMaxScaler**.
  
3️⃣ **Xây dựng và huấn luyện mô hình**:
- Huấn luyện các mô hình **K-Nearest Neighbors (KNN)** và **Hồi quy Logistic (Logistic Regression)**.
  
4️⃣ **Đánh giá mô hình**:
- So sánh hiệu suất của hai mô hình dựa trên các độ đo Accuracy và F1-score.

### 🏆 Kết quả
| Mô hình | Độ chính xác (Accuracy) | Macro avg F1-score | Nhận xét |
| :--- | :--- | :--- | :--- |
| **KNN** | **99%** | **0.99** | Hiệu suất vượt trội, chỉ dự đoán sai 4 mẫu trên tổng 400 mẫu kiểm tra. |
| Logistic Regression | 98% | 0.98 | Hoạt động ổn định và **dễ giải thích**. |

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`LogisticRegression`, `KNeighborsClassifier`, `MinMaxScaler`)
*   `imblearn` (`RandomOverSampler`)

---

## 💎 Dự án 4: Dự đoán giá kim cương

### 🎯 Giới thiệu
Dự án hồi quy này nhằm mục đích xây dựng một mô hình dự đoán giá bán của viên kim cương (biến liên tục) dựa trên các thuộc tính vật lý và chất lượng của nó (hơn 53.000 mẫu dữ liệu).

### ⚙️ Quy trình thực hiện
1️⃣ **Khám phá dữ liệu (EDA)**:
- Loại bỏ **149 dòng dữ liệu trùng lặp** và các dòng có giá trị 0 ở các cột kích thước (`x`, `y`, `z`).
  
2️⃣ **Tiền xử lý dữ liệu**:
- Mã hóa các biến phân loại (`cut`, `color`, `clarity`) thành số bằng **LabelEncoder**.
- Xử lý giá trị ngoại lệ (outlier) ở biến `carat` bằng phương pháp **IQR** và thay thế bằng giá trị trung bình.
- **Chuẩn hóa** tất cả các biến đầu vào và biến mục tiêu (`price`) bằng **MinMaxScaler**.
  
3️⃣ **Xây dựng và huấn luyện mô hình**:
- Huấn luyện ba mô hình hồi quy: **Linear Regression**, **Random Forest Regressor**, và **Decision Tree Regressor**.
  
4️⃣ **Đánh giá mô hình**:
- Đánh giá các mô hình bằng các độ đo: **R² Score**, Mean Absolute Error (MAE), và Root Mean Squared Error (RMSE).

### 🏆 Kết quả
| Mô hình | Hệ số xác định R² | MAE | RMSE | Nhận xét |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9784** | **0.0151** | **0.0318** | **Hiệu suất vượt trội nhất**, giải thích được gần 98% sự biến động của giá kim cương. |
| Decision Tree | 0.9616 | 0.0201 | 0.0423 | - |
| Linear Regression | 0.8129 | - | - | Hiệu quả thấp nhất. |

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`, `LabelEncoder`, `MinMaxScaler`)

---

## 📈 Dự án 5: Dự đoán giá cổ phiếu Uber

### 🎯 Giới thiệu
Dự án này phân tích và xây dựng mô hình **chuỗi thời gian** để dự đoán giá đóng cửa của cổ phiếu Uber trong tương lai gần.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**:
- Sử dụng giá đóng cửa điều chỉnh (`Adj Close`).
- Biến đổi **logarit** để làm mượt và ổn định phương sai.

2️⃣ **Phân tích chuỗi thời gian**:
- Kiểm định tính dừng của chuỗi thời gian bằng kiểm định ADF và KPSS (chuỗi gốc **chưa dừng**).
- Thực hiện **sai phân bậc 1 (d=1)** để chuỗi đạt tính dừng.
- Phân tích PACF gợi ý tham số $p=2$.

3️⃣ **Xây dựng mô hình ARIMA**:
- Tham số tối ưu được xác định tự động bằng `auto_arima()` là **ARIMA(2,1,2)**.
- Huấn luyện mô hình ARIMA trên tập huấn luyện.

4️⃣ **Dự đoán và đánh giá**:
- Đánh giá mô hình bằng chỉ số **RMSE** trên tập kiểm tra.
- So sánh với một mô hình cơ sở (baseline) dự đoán bằng giá trị trung bình.

### 🏆 Kết quả
| Mô hình | RMSE (Tập kiểm tra) | Nhận xét |
| :--- | :--- | :--- |
| **ARIMA(2,1,2)** | **0.115** | Mô hình ARIMA **vượt trội hơn đáng kể** so với Baseline. |
| Baseline | 0.643 | - |

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `statsmodels`
*   `pmdarima` (`auto_arima`)

---

## 🍊 Dự án 6: Phân loại chất lượng cam bằng CNN

### 🎯 Giới thiệu
Dự án này ứng dụng **Mạng Nơ-ron Tích chập (CNN)** để xây dựng một mô hình phân loại hình ảnh, có khả năng phân biệt giữa cam chất lượng tốt và cam chất lượng kém (Phân loại nhị phân hình ảnh).

### ⚙️ Quy trình thực hiện
1️⃣ **Chuẩn bị dữ liệu**:
- Đọc, **resize ảnh về kích thước chuẩn (32x32)**.
- Chuẩn hóa giá trị pixel về khoảng $[0, 1]$.
- Sử dụng **StratifiedShuffleSplit** để chia tập huấn luyện và validation, đảm bảo tỷ lệ lớp cân bằng.

2️⃣ **Xây dựng mô hình CNN**:
- Kiến trúc gồm 2 lớp tích chập (Conv2D), MaxPooling và các lớp Dense kết hợp **Dropout (50%)** để giảm overfitting.

3️⃣ **Huấn luyện mô hình**:
- Biên dịch và huấn luyện mô hình.

4️⃣ **Đánh giá và xuất kết quả**:
- Đánh giá hiệu suất trên tập kiểm tra thực tế.
- Phân tích sự thiên lệch của mô hình.

### 🏆 Kết quả
| Hiệu suất | Độ chính xác (Test) | Recall lớp 0 (Cam xấu) | Recall lớp 1 (Cam tốt) |
| :--- | :--- | :--- | :--- |
| **Trên Tập Kiểm tra** | **72.5%** | 1.00 | 0.45 |
| **Đánh giá thiên lệch** | - | Mô hình thiên lệch về lớp "cam xấu" (nhận diện rất tốt) nhưng bỏ sót nhiều cam tốt. |

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`
*   `opencv-python`
*   `tensorflow`, `keras`
*   `scikit-learn`
