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
Dự án này tập trung vào việc xây dựng một mô hình phân loại để dự đoán điểm tín dụng của khách hàng (Poor, Standard, Good) dựa trên các thông tin cá nhân và lịch sử tài chính của họ (bài toán phân loại đa lớp).

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**: Tải tập dữ liệu train và test, sau đó gộp lại để thực hiện tiền xử lý đồng bộ.  
2️⃣ **Làm sạch và tiền xử lý dữ liệu**:  
- Thực hiện làm sạch văn bản (cột `Type_of_Loan`).
- Chuyển đổi kiểu dữ liệu của các cột cho phù hợp (ví dụ: ID, tuổi, thu nhập).  
- Xử lý các giá trị thiếu (`NaN`) bằng mode và các giá trị đặc biệt.  
- Xử lý các giá trị ngoại lệ (outlier) bằng phương pháp IQR (thay bằng trung bình).  
- Điền các giá trị thiếu cho các cột số bằng giá trị trung bình và các cột phân loại bằng giá trị mode.

3️⃣ **Kỹ thuật đặc trưng (Feature Engineering)**:  
- Tạo biến giả (dummy variables) từ các cột phân loại.
- Sử dụng `OrdinalEncoder` để chuyển đổi các cột phân loại còn lại thành dạng số.

4️⃣ **Lựa chọn và chuẩn hóa đặc trưng**:  
- Loại bỏ các đặc trưng không cần thiết dựa trên phân tích.  
- Chuẩn hóa dữ liệu bằng `MinMaxScaler`.
  
5️⃣ **Xây dựng và huấn luyện mô hình**:  
- Phân chia dữ liệu thành tập huấn luyện và tập xác thực.  
- Huấn luyện các mô hình `SVC` và `KNeighborsClassifier`.
  
6️⃣ **Đánh giá mô hình**:  
- Đánh giá hiệu suất của các mô hình trên tập xác thực bằng `classification_report` và `confusion_matrix`.  

### 🏆 Kết quả
> - **K-Nearest Neighbors (KNN):** Đạt hiệu suất tốt nhất với **Độ chính xác (Accuracy) tổng thể 71%**. F1-score cho các lớp dao động từ 0.61 (Poor) đến 0.75 (Good), cho thấy khả năng phân loại đồng đều và ổn định.
> - **Support Vector Machine (SVM):** Hiệu suất kém hơn đáng kể (Accuracy 51%). Mô hình này đặc biệt yếu trong việc nhận diện lớp 'Poor' (F1-score chỉ đạt 0.03).

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`KNeighborsClassifier`, `SVC`, `GridSearchCV`, `MinMaxScaler`, `OrdinalEncoder`)
*   `statsmodels`

---

## 👥 Dự án 2: Phân cụm khách hàng ngân hàng

### 🎯 Giới thiệu
Dự án này sử dụng các kỹ thuật phân cụm (học không giám sát) để phân nhóm khách hàng của một ngân hàng dựa trên hành vi giao dịch và thông tin nhân khẩu học. Mục tiêu là để xác định các phân khúc khách hàng khác nhau nhằm phục vụ cho các chiến lược kinh doanh.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu (EDA)**: Tải và kiểm tra thông tin, dữ liệu thiếu, dữ liệu trùng lặp. Sử dụng tập mẫu 100.000 bản ghi để huấn luyện.
2️⃣ **Làm sạch dữ liệu**:
- Loại bỏ các hàng có giá trị thiếu.
- Xử lý các giá trị không hợp lệ (ví dụ: tuổi âm, tuổi > 100).
- Chuyển đổi cột ngày tháng sang định dạng datetime và tính toán tuổi (Age) của khách hàng.

3️⃣ **Phân tích RFM (Recency, Frequency, Monetary)**:
- Tính toán các giá trị Recency, Frequency, và Monetary cho mỗi khách hàng.
- Kết hợp các chỉ số RFM vào bộ dữ liệu chính.
  
4️⃣ **Trực quan hóa dữ liệu**: Sử dụng `matplotlib` và `seaborn` để vẽ các biểu đồ phân tích (boxplot, histogram, bar chart) nhằm hiểu rõ hơn về đặc điểm của dữ liệu.   
5️⃣ **Tiền xử lý nâng cao**:
- Loại bỏ các giá trị ngoại lệ (outlier) bằng IQR.
- Xử lý các giá trị thiếu còn lại bằng cách điền giá trị trung vị (median).
- Chuẩn hóa dữ liệu bằng `StandardScaler`.
  
6️⃣ **Xây dựng mô hình phân cụm**:
- Sử dụng phương pháp Elbow và Silhouette để xác định số cụm tối ưu.
- Áp dụng thuật toán **K-Means** và **DBSCAN**.
- Sử dụng **PCA** để giảm chiều dữ liệu xuống 3 thành phần chính và trực quan hóa các cụm.
  
7️⃣ **Đánh giá và phân tích cụm**:
- Đánh giá mô hình bằng chỉ số Silhouette.
- Trực quan hóa các cụm và phân tích đặc điểm của từng cụm.

### 🏆 Kết quả
> - **K-Means + PCA (k=5):** Đạt **Silhouette Score = 0.2956**, cao hơn so với K-Means gốc (0.2544). Đây là mô hình được đề xuất để triển khai thực tế do khả năng phân cụm rõ ràng.
> - **DBSCAN:** Đạt Silhouette Score = 0.3608 (cao hơn), nhưng đã xác định tới **39.9%** dữ liệu là nhiễu (outlier). Mô hình này hữu ích hơn cho mục đích phát hiện các khách hàng có hành vi bất thường.

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`, `plotly`
*   `scikit-learn` (`KMeans`, `DBSCAN`, `StandardScaler`, `PCA`)
*   `kneed`

---

## 💵 Dự án 3: Phân loại tiền thật/giả

### 🎯 Giới thiệu
Mục tiêu của dự án này là xây dựng một mô hình có khả năng phân biệt giữa tiền thật và tiền giả (phân loại nhị phân) dựa trên các đặc trưng đo lường vật lý (chiều dài, chiều cao, lề).

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**: Tải dữ liệu (1.500 mẫu) và kiểm tra các thông tin cơ bản, xử lý các giá trị thiếu (điền mode cho `margin_low`).
2️⃣ **Tiền xử lý dữ liệu**:
- Chuyển đổi biến mục tiêu `is_genuine` thành dạng số (0/1).
- Kiểm tra sự mất cân bằng của dữ liệu (1000 thật / 500 giả) và sử dụng kỹ thuật `RandomOverSampler` để cân bằng lại lớp thiểu số.
- Chuẩn hóa các đặc trưng bằng `MinMaxScaler`.
  
3️⃣ **Xây dựng và huấn luyện mô hình**:
- Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra.
- Sử dụng `GridSearchCV` để tìm ra tham số `n_neighbors` tốt nhất cho mô hình **KNN**.
- Sử dụng `GridSearchCV` để tìm các tham số tối ưu cho mô hình **Logistic Regression**.
  
4️⃣ **Đánh giá mô hình**:
- So sánh hiệu suất của hai mô hình dựa trên `classification_report` và `confusion_matrix`.

### 🏆 Kết quả
> - **K-Nearest Neighbors (KNN):** Đạt hiệu suất vượt trội với **Độ chính xác (Accuracy) 99%** và Macro F1-score 0.99. Trên tập kiểm tra 400 mẫu, mô hình chỉ dự đoán sai 4 mẫu.
> - **Hồi quy Logistic (Logistic Regression):** Đạt Accuracy 98% và F1-score 0.98. Mặc dù thấp hơn KNN một chút, đây là mô hình ổn định và dễ diễn giải.

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`LogisticRegression`, `KNeighborsClassifier`, `GridSearchCV`, `MinMaxScaler`)
*   `imblearn` (`RandomOverSampler`)

---

## 💎 Dự án 4: Dự đoán giá kim cương

### 🎯 Giới thiệu
Dự án hồi quy này nhằm mục đích xây dựng một mô hình dự đoán giá (price) của kim cương dựa trên các thuộc tính vật lý của nó như carat, cut, color, clarity, v.v.

### ⚙️ Quy trình thực hiện
1️⃣ **Khám phá dữ liệu (EDA)**:
- Tải và kiểm tra thông tin chung của dữ liệu (hơn 53.000 mẫu).
- Phân tích mối tương quan và trực quan hóa dữ liệu để phát hiện outlier.
  
2️⃣ **Tiền xử lý dữ liệu**:
- Loại bỏ 149 hàng dữ liệu bị trùng lặp.
- Xử lý các giá trị không hợp lệ (kích thước x, y, z bằng 0).
- Mã hóa các biến phân loại (`cut`, `color`, `clarity`) bằng `LabelEncoder`.
- Xử lý outlier ở biến `carat` bằng phương pháp IQR (thay bằng giá trị trung bình).
- Kiểm tra đa cộng tuyến bằng hệ số VIF.
  
3️⃣ **Chuẩn hóa và xây dựng mô hình**:
- Chuẩn hóa tất cả biến đầu vào và biến mục tiêu (`price`) bằng `MinMaxScaler`.
- Phân chia dữ liệu thành các tập huấn luyện và kiểm tra.
- Xây dựng và huấn luyện ba mô hình hồi quy: **Linear Regression**, **Random Forest Regressor**, và **Decision Tree Regressor**.
  
4️⃣ **Đánh giá mô hình**:
- Đánh giá các mô hình bằng các độ đo: R² Score, Mean Absolute Error (MAE), và Root Mean Squared Error (RMSE).
- So sánh hiệu suất của các mô hình và chọn ra mô hình tốt nhất.

### 🏆 Kết quả
> - **Random Forest Regressor:** Đạt hiệu suất vượt trội nhất, giải thích được gần 98% sự biến động của giá.
>   - **R² Score:** 0.9784
>   - **MAE:** 0.0151
>   - **RMSE:** 0.0318
> - **Decision Tree Regressor:** Hiệu suất tốt (R² = 0.9616, MAE = 0.0201, RMSE = 0.0423).
> - **Linear Regression:** Hiệu quả thấp nhất (R² = 0.8129).

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`, `LabelEncoder`, `MinMaxScaler`)
*   `statsmodels`

---

## 📈 Dự án 5: Dự đoán giá cổ phiếu Uber

### 🎯 Giới thiệu
Dự án này phân tích và xây dựng mô hình chuỗi thời gian ARIMA để dự đoán giá đóng cửa của cổ phiếu Uber (từ 2019 đến 2025).

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**:
- Tải dữ liệu (giá `Adj Close`), chuyển đổi cột `Date` sang định dạng datetime.
- Trực quan hóa giá cổ phiếu theo thời gian.

2️⃣ **Phân tích chuỗi thời gian**:
- Phân rã chuỗi thời gian (`seasonal_decompose`).
- Biến đổi logarit để ổn định phương sai.
- Kiểm định tính dừng của chuỗi (ADF và KPSS) -> Chuỗi chưa dừng.
- Thực hiện sai phân bậc 1 (d=1) để làm cho chuỗi thời gian trở nên dừng.
- Sử dụng biểu đồ PACF để gợi ý tham số `p` (p=2).

3️⃣ **Xây dựng mô hình ARIMA**:
- Sử dụng `auto_arima` để tự động tìm ra các tham số (p, d, q) tối ưu.
- Huấn luyện mô hình ARIMA trên tập huấn luyện.

4️⃣ **Dự đoán và đánh giá**:
- Dự đoán giá trị trên tập kiểm tra.
- Đánh giá mô hình bằng chỉ số RMSE và so sánh với một mô hình cơ sở (baseline - dự đoán bằng giá trị trung bình).
- Trực quan hóa kết quả dự đoán so với giá trị thực tế.

### 🏆 Kết quả
> - **Mô hình tối ưu:** `ARIMA(2,1,2)` (xác định qua `auto_arima` sau khi sai phân bậc 1).
> - **Hiệu suất (Tập Test):** Đạt **RMSE = 0.115**.
> - **So sánh:** Mô hình ARIMA hoạt động hiệu quả hơn đáng kể so với mô hình baseline (dự đoán bằng giá trị trung bình) vốn có RMSE = 0.643.

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `statsmodels` (`ARIMA`, `seasonal_decompose`, `adfuller`, `kpss`)
*   `pmdarima` (`auto_arima`)

---

## 🍊 Dự án 6: Phân loại chất lượng cam bằng CNN

### 🎯 Giới thiệu
Dự án này ứng dụng Mạng Nơ-ron Tích chập (CNN) để xây dựng một mô hình phân loại hình ảnh, có khả năng phân biệt giữa cam chất lượng tốt (Good) và cam chất lượng kém (Bad).

### ⚙️ Quy trình thực hiện
1️⃣ **Chuẩn bị dữ liệu**:
- Tải và sắp xếp dữ liệu hình ảnh (hơn 2.000 ảnh train, 400 ảnh test) vào các thư mục tương ứng.
- Đọc hình ảnh, chuyển đổi kích thước về (32x32), và chuẩn hóa giá trị pixel về khoảng [0, 1].
- Tạo các tập dữ liệu huấn luyện và kiểm tra.

2️⃣ **Phân chia dữ liệu**:
- Sử dụng `StratifiedShuffleSplit` để chia tập huấn luyện thành tập huấn luyện nhỏ hơn và tập xác thực, đảm bảo sự cân bằng về tỷ lệ các lớp.

3️⃣ **Xây dựng mô hình CNN**:
- Thiết kế kiến trúc mô hình CNN bao gồm 2 lớp `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, và `Dropout` (50%) để giảm overfitting.
- Lớp đầu ra sử dụng hàm kích hoạt `sigmoid` cho bài toán phân loại nhị phân.

4️⃣ **Huấn luyện mô hình**:
- Biên dịch mô hình với hàm mất mát `sparse_categorical_crossentropy` và trình tối ưu hóa `Adam`.
- Sử dụng `EarlyStopping` để tránh overfitting.
- Huấn luyện mô hình.

5️⃣ **Đánh giá và xuất kết quả**:
- Đánh giá độ chính xác của mô hình trên tập kiểm tra.
- Trực quan hóa lịch sử huấn luyện (độ chính xác và mất mát).

### 🏆 Kết quả
> - **Hiệu suất mô hình (CNN):** Đạt **Độ chính xác (Accuracy) 72.5%** trên tập kiểm tra (Test set).
> - **Đánh giá (Overfitting):** Mặc dù độ chính xác trên tập validation rất cao (>98%), hiệu suất trên tập test thực tế thấp hơn đáng kể, cho thấy dấu hiệu của overfitting.
> - **Thiên lệch (Bias):** Mô hình bị thiên lệch nặng, có xu hướng dự đoán là "cam xấu". Mô hình nhận diện được 100% cam xấu (Recall lớp 0 = 1.00) nhưng lại bỏ sót rất nhiều cam tốt (Recall lớp 1 = 0.45).

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`
*   `opencv-python`
*   `tensorflow`, `keras`
*   `scikit-learn` (`StratifiedShuffleSplit`)
