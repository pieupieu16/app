import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# --- BƯỚC 1: TẢI DỮ LIỆU (Step 1: Load Data) ---
# Chúng ta sử dụng file parquet để đạt hiệu suất cao nhất
data_path = 'processed_housing_data.parquet'

if not os.path.exists(data_path):
    print(f"Lỗi: Không tìm thấy file {data_path}!")
    # Nếu chưa có parquet, bạn có thể đổi thành .csv tạm thời
    # df = pd.read_csv('VN_housing_dataset.csv')
else:
    df = pd.read_parquet(data_path)
    print("Đã tải dữ liệu thành công! (Data loaded successfully!)")

# --- BƯỚC 2: CHUẨN BỊ DỮ LIỆU (Step 2: Data Preparation) ---
# Tách biến mục tiêu (Target) và các đặc trưng (Features)
# Giả sử cột giá nhà của bạn tên là 'Giá nhà'
TARGET = 'Giá nhà'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Lưu danh sách các cột để app Streamlit biết thứ tự input
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

# Chia dữ liệu để kiểm tra (Train/Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH (Step 3: Model Training) ---
print("Đang huấn luyện mô hình Random Forest... (Training model...)")
model = RandomForestRegressor(
    n_estimators=100,      # Số lượng cây trong rừng
    max_depth=15,          # Độ sâu tối đa của cây để tránh quá khớp (Overfitting)
    random_state=42,
    n_jobs=-1              # Sử dụng toàn bộ nhân CPU để chạy nhanh hơn
)

model.fit(X_train, y_train)

# --- BƯỚC 4: ĐÁNH GIÁ (Step 4: Evaluation) ---
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Độ lỗi trung bình (MAE): {error:.2f} Tỷ")

# --- BƯỚC 5: LƯU MÔ HÌNH (Step 5: Save Model) ---
# QUAN TRỌNG: Tắt OneDrive trước khi chạy dòng này
model_filename = 'house_price_model.pkl'
joblib.dump(model, model_filename, compress=3) # Nén mức 3 để file nhỏ hơn
print(f"Đã lưu mô hình mới tại: {model_filename}")