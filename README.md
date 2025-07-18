# 🤖 Tổng Hợp Các Dự Án Machine Learning Trong Môn Học "Máy Học Thống Kê"

👋 Chào mừng bạn đến với kho lưu trữ các dự án trong đồ án môn học 'Máy học thông kê". Kho này chứa một loạt các dự án thể hiện những kỹ năng và kiến thức của tôi trong lĩnh vực học máy, từ xử lý dữ liệu đến xây dựng và đánh giá các mô hình phức tạp.

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
Dự án này tập trung vào việc xây dựng một mô hình phân loại để dự đoán điểm tín dụng của khách hàng (Tốt, Khá, Kém) dựa trên các thông tin cá nhân và lịch sử tài chính của họ.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**: Tải tập dữ liệu train và test, sau đó gộp lại để thực hiện tiền xử lý đồng bộ.  
2️⃣ **Làm sạch và tiền xử lý dữ liệu**:  
- Thực hiện làm sạch văn bản để loại bỏ các ký tự không cần thiết.  
- Chuyển đổi kiểu dữ liệu của các cột cho phù hợp (ví dụ: ID, tuổi, thu nhập).  
    *   Xử lý các giá trị thiếu (`NaN`) và các giá trị đặc biệt.  
    *   Xử lý các giá trị ngoại lệ (outlier) bằng phương pháp IQR.  
    *   Điền các giá trị thiếu cho các cột số bằng giá trị trung bình và các cột phân loại bằng giá trị mode.\   
3️⃣ **Kỹ thuật đặc trưng (Feature Engineering)**:  
    *   Tạo biến giả (dummy variables) từ các cột phân loại có chứa nhiều giá trị, ví dụ như `Type_of_Loan`.  
    *   Sử dụng `OrdinalEncoder` để chuyển đổi các cột phân loại còn lại thành dạng số.  
4️⃣ **Lựa chọn và chuẩn hóa đặc trưng**:  
    *   Loại bỏ các đặc trưng không cần thiết dựa trên phân tích.  
    *   Chuẩn hóa dữ liệu bằng `MinMaxScaler`.  
5️⃣ **Xây dựng và huấn luyện mô hình**:  
    *   Phân chia dữ liệu thành tập huấn luyện và tập xác thực.  
    *   Sử dụng `GridSearchCV` để tìm các tham số tối ưu cho mô hình `SVC`.  
    *   Huấn luyện các mô hình `SVC` và `KNeighborsClassifier`.  
6️⃣ **Đánh giá mô hình**:  
    *   Đánh giá hiệu suất của các mô hình trên tập xác thực bằng `classification_report` và `confusion_matrix`.  

### 🏆 Kết quả
> *(Phần này để trống để bạn tự điền kết quả đánh giá mô hình)*

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`LogisticRegression`, `KNeighborsClassifier`, `SVC`, `GridSearchCV`, `MinMaxScaler`)
*   `statsmodels`

---

## 👥 Dự án 2: Phân cụm khách hàng ngân hàng

### 🎯 Giới thiệu
Dự án này sử dụng các kỹ thuật phân cụm để phân nhóm khách hàng của một ngân hàng dựa trên hành vi giao dịch và thông tin nhân khẩu học. Mục tiêu là để xác định các phân khúc khách hàng khác nhau nhằm phục vụ cho các chiến lược kinh doanh.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu (EDA)**: Tải và kiểm tra thông tin, dữ liệu thiếu, dữ liệu trùng lặp.
2️⃣ **Làm sạch dữ liệu**:
    *   Loại bỏ các hàng có giá trị thiếu.
    *   Xử lý các giá trị không hợp lệ trong cột giới tính.
    *   Chuyển đổi cột ngày tháng sang định dạng datetime và tính toán tuổi của khách hàng.
3️⃣ **Phân tích RFM (Recency, Frequency, Monetary)**:
    *   Tính toán các giá trị Recency, Frequency, và Monetary cho mỗi khách hàng.
    *   Kết hợp các chỉ số RFM vào bộ dữ liệu chính.
4️⃣ **Trực quan hóa dữ liệu**: Sử dụng `matplotlib` và `seaborn` để vẽ các biểu đồ phân tích (boxplot, histogram, bar chart) nhằm hiểu rõ hơn về đặc điểm của dữ liệu.
5️⃣ **Tiền xử lý nâng cao**:
    *   Loại bỏ các giá trị ngoại lệ (outlier).
    *   Xử lý các giá trị thiếu còn lại bằng cách điền giá trị trung vị (median) hoặc các phương pháp phù hợp khác.
    *   Chuẩn hóa dữ liệu bằng `StandardScaler`.
6️⃣ **Xây dựng mô hình phân cụm**:
    *   Sử dụng phương pháp Elbow và Silhouette để xác định số cụm tối ưu cho **K-Means**.
    *   Áp dụng thuật toán **K-Means** và **DBSCAN**.
    *   Sử dụng **PCA** để giảm chiều dữ liệu và trực quan hóa các cụm.
7️⃣ **Đánh giá và phân tích cụm**:
    *   Đánh giá mô hình bằng chỉ số Silhouette.
    *   Trực quan hóa các cụm trong không gian 3D (sử dụng `plotly`) và phân tích đặc điểm của từng cụm.

### 🏆 Kết quả
> *(Phần này để trống để bạn tự điền kết quả phân tích các cụm khách hàng)*

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`, `plotly`
*   `scikit-learn` (`KMeans`, `DBSCAN`, `StandardScaler`, `PCA`)
*   `kneed`

---

## 💵 Dự án 3: Phân loại tiền thật/giả

### 🎯 Giới thiệu
Mục tiêu của dự án này là xây dựng một mô hình có khả năng phân biệt giữa tiền thật và tiền giả dựa trên các đặc trưng được trích xuất từ hình ảnh của tờ tiền.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**: Tải dữ liệu và kiểm tra các thông tin cơ bản, xử lý các giá trị thiếu.
2️⃣ **Tiền xử lý dữ liệu**:
    *   Chuyển đổi biến mục tiêu `is_genuine` thành dạng số (0/1).
    *   Kiểm tra sự mất cân bằng của dữ liệu và sử dụng kỹ thuật `RandomOverSampler` để cân bằng lại lớp thiểu số.
    *   Chuẩn hóa các đặc trưng bằng `MinMaxScaler` do có sự chênh lệch về thang đo.
3️⃣ **Xây dựng và huấn luyện mô hình**:
    *   Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra.
    *   Sử dụng `GridSearchCV` để tìm ra tham số `n_neighbors` tốt nhất cho mô hình **KNN**.
    *   Sử dụng `GridSearchCV` để tìm các tham số tối ưu cho mô hình **Logistic Regression**.
4️⃣ **Đánh giá mô hình**:
    *   So sánh hiệu suất của hai mô hình dựa trên `classification_report` và `confusion_matrix`.
    *   Đưa ra nhận xét về sự phù hợp của từng mô hình, đặc biệt là khả năng dự đoán đúng các trường hợp tiền giả (False Negative).

### 🏆 Kết quả
> *(Phần này để trống để bạn tự điền kết quả so sánh và nhận xét mô hình)*

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`LogisticRegression`, `KNeighborsClassifier`, `GridSearchCV`, `MinMaxScaler`)
*   `imblearn` (`RandomOverSampler`)

---

## 💎 Dự án 4: Dự đoán giá kim cương

### 🎯 Giới thiệu
Dự án hồi quy này nhằm mục đích xây dựng một mô hình dự đoán giá của kim cương dựa trên các thuộc tính vật lý của nó như carat, cut, color, clarity, v.v.

### ⚙️ Quy trình thực hiện
1️⃣ **Khám phá dữ liệu (EDA)**:
    *   Tải và kiểm tra thông tin chung của dữ liệu.
    *   Phân tích mối tương quan giữa các biến số.
    *   Trực quan hóa dữ liệu để phát hiện outlier (sử dụng boxplot) và kiểm tra phân phối của các biến (sử dụng histogram).
2️⃣ **Tiền xử lý dữ liệu**:
    *   Loại bỏ các hàng dữ liệu bị trùng lặp.
    *   Xử lý các giá trị không hợp lệ (ví dụ: kích thước x, y, z bằng 0).
    *   Mã hóa các biến phân loại (`cut`, `color`, `clarity`) bằng `LabelEncoder`.
    *   Xử lý outlier bằng cách thay thế chúng bằng giá trị trung bình.
    *   Kiểm tra đa cộng tuyến bằng hệ số VIF.
3️⃣ **Chuẩn hóa và xây dựng mô hình**:
    *   Chuẩn hóa dữ liệu bằng `MinMaxScaler`.
    *   Phân chia dữ liệu thành các tập huấn luyện và kiểm tra.
    *   Xây dựng và huấn luyện ba mô hình hồi quy: **Linear Regression**, **Random Forest Regressor**, và **Decision Tree Regressor**.
4️⃣ **Đánh giá mô hình**:
    *   Đánh giá các mô hình bằng các độ đo: R² Score, Mean Absolute Error (MAE), và Root Mean Squared Error (RMSE).
    *   Sử dụng phương pháp K-Fold Cross Validation để đánh giá độ ổn định của mô hình.
    *   So sánh hiệu suất của các mô hình và chọn ra mô hình tốt nhất.

### 🏆 Kết quả
> *(Phần này để trống để bạn tự điền kết quả đánh giá và so sánh các mô hình)*

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`, `LabelEncoder`, `MinMaxScaler`)
*   `statsmodels`

---

## 📈 Dự án 5: Dự đoán giá cổ phiếu Uber

### 🎯 Giới thiệu
Dự án này phân tích và xây dựng mô hình chuỗi thời gian để dự đoán giá đóng cửa của cổ phiếu Uber.

### ⚙️ Quy trình thực hiện
1️⃣ **Tải và khám phá dữ liệu**:
    *   Tải dữ liệu và chuyển đổi cột `Date` sang định dạng datetime.
    *   Trực quan hóa giá cổ phiếu theo thời gian để nhận diện xu hướng chung và các biến động.
2️⃣ **Phân tích chuỗi thời gian**:
    *   Phân rã chuỗi thời gian (`seasonal_decompose`) để xem các thành phần xu hướng, mùa vụ và phần dư.
    *   Sử dụng biểu đồ ACF và PACF để xác định các tham số tự tương quan.
    *   Kiểm định tính dừng của chuỗi thời gian bằng kiểm định Dickey-Fuller (ADF) và KPSS.
    *   Thực hiện sai phân để làm cho chuỗi thời gian trở nên dừng.
3️⃣ **Xây dựng mô hình ARIMA**:
    *   Sử dụng `auto_arima` để tự động tìm ra các tham số (p, d, q) tối ưu cho mô hình ARIMA.
    *   Huấn luyện mô hình ARIMA trên tập huấn luyện.
4️⃣ **Dự đoán và đánh giá**:
    *   Dự đoán giá trị trên tập kiểm tra.
    *   Đánh giá mô hình bằng các chỉ số RMSE và so sánh với một mô hình cơ sở (baseline).
    *   Trực quan hóa kết quả dự đoán so với giá trị thực tế.

### 🏆 Kết quả
> *(Phần này để trống để bạn tự điền kết quả dự đoán và đánh giá mô hình ARIMA)*

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`, `seaborn`
*   `statsmodels` (`ARIMA`, `seasonal_decompose`, `adfuller`, `kpss`)
*   `pmdarima` (`auto_arima`)

---

## 🍊 Dự án 6: Phân loại chất lượng cam bằng CNN

### 🎯 Giới thiệu
Dự án này ứng dụng Mạng Nơ-ron Tích chập (CNN) để xây dựng một mô hình phân loại hình ảnh, có khả năng phân biệt giữa cam chất lượng tốt và cam chất lượng kém.

### ⚙️ Quy trình thực hiện
1️⃣ **Chuẩn bị dữ liệu**:
    *   Tải và sắp xếp dữ liệu hình ảnh vào các thư mục tương ứng với hai lớp: `Orange_bad` và `Orange_good`.
    *   Đọc hình ảnh, chuyển đổi kích thước về (32x32), và chuẩn hóa giá trị pixel.
    *   Tạo các tập dữ liệu huấn luyện và kiểm tra.
2️⃣ **Phân chia dữ liệu**:
    *   Sử dụng `StratifiedShuffleSplit` để chia tập huấn luyện thành tập huấn luyện nhỏ hơn và tập xác thực, đảm bảo sự cân bằng về tỷ lệ các lớp.
3️⃣ **Xây dựng mô hình CNN**:
    *   Thiết kế kiến trúc mô hình CNN bao gồm các lớp `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, và `Dropout`.
    *   Lớp đầu ra sử dụng hàm kích hoạt `sigmoid` cho bài toán phân loại nhị phân.
4️⃣ **Huấn luyện mô hình**:
    *   Biên dịch mô hình với hàm mất mát `sparse_categorical_crossentropy` và trình tối ưu hóa `Adam`.
    *   Sử dụng `EarlyStopping` để tránh overfitting.
    *   Huấn luyện mô hình trên tập dữ liệu đã chuẩn bị.
5️⃣ **Đánh giá và xuất kết quả**:
    *   Đánh giá độ chính xác của mô hình trên tập kiểm tra.
    *   Trực quan hóa lịch sử huấn luyện (độ chính xác và mất mát).
    *   Dự đoán nhãn cho các ảnh trong tập kiểm tra và lưu kết quả vào file CSV.

### 🏆 Kết quả
> *(Phần này để trống để bạn tự điền độ chính xác và các kết quả khác của mô hình CNN)*

### 💻 Công nghệ sử dụng
*   `pandas`, `numpy`
*   `matplotlib`
*   `opencv-python`
*   `tensorflow`, `keras`
*   `scikit-learn`
