import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# --- 1. TẢI DỮ LIỆU (Load Data) ---
# Sử dụng file parquet để tốc độ nhanh hơn
data_path = 'processed_housing_data.parquet'

df = pd.read_parquet(data_path)

# --- 2. CHUẨN BỊ DỮ LIỆU (Prepare Data) ---
# Loại bỏ các dòng bị thiếu giá trị (Handling missing values)
df = df.dropna()

# Tách Features (X) và Target (y)
# Giả sử tên cột giá là 'Giá nhà'
X = df.drop(columns=['Giá nhà'])
y = df['Giá nhà']

# [QUAN TRỌNG] Lưu danh sách các cột ngay tại đây
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("✅ Đã lưu file: model_columns.pkl")

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. HUẤN LUYỆN (Training) ---
print("🚀 Đang huấn luyện mô hình Random Forest...")
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=12)
model.fit(X_train, y_train)

# --- 4. LƯU MÔ HÌNH (Save Model) ---
# Lưu file pkl chính cho bộ não AI
joblib.dump(model, 'house_price_model.pkl', compress=3)
print("✅ Đã lưu file: house_price_model.pkl")

print("\n--- HOÀN THÀNH (FINISHED) ---")
print("Vui lòng kiểm tra thư mục, bạn sẽ thấy 2 file mới xuất hiện.")