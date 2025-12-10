import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. LOAD DỮ LIỆU
# ==========================================
# Đọc file CSV (Giả sử bạn đã giải nén file zip)
# Nếu file vẫn trong zip, dùng: pd.read_csv('processed_housing_data.zip')
try:
    df = pd.read_csv('processed_housing_data.zip')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'processed_housing_data.csv'. Hãy kiểm tra lại đường dẫn.")
    exit()

print("Dữ liệu gốc:", df.shape)

# ==========================================
# 2. TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ==========================================

# Bước 1: Loại bỏ các cột không cần thiết cho việc dự báo
# 'STT': Chỉ là số thứ tự
# 'Địa chỉ': Quá chi tiết, gộp chung với Quận/Huyện rồi nên bỏ để tránh nhiễu
cols_to_drop = ['STT', 'Địa chỉ']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Bước 2: Tách Features (X) và Target (y)
target_col = 'Giá nhà'  # Tên cột giá trong file của bạn
if target_col not in df.columns:
    raise ValueError(f"Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Bước 3: Xác định nhóm cột Số và Cột Chữ
# Lưu ý: 'Năm' và 'Tháng' được đưa vào nhóm số (hoặc phân loại) để mô hình học
numeric_features = ['Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng', 'Năm', 'Tháng']
categorical_features = ['Quận', 'Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý']

# Kiểm tra xem các cột này có thực sự tồn tại trong file không
numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]

print(f"Features số lượng: {len(numeric_features)} cột")
print(f"Features phân loại: {len(categorical_features)} cột")

# Bước 4: Chia tập Train/Test (80% học, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. THIẾT LẬP PIPELINE
# ==========================================

# Xử lý cột số: Điền giá trị thiếu (nếu có) -> Chuẩn hóa Scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Xử lý cột chữ: Điền giá trị thiếu -> OneHotEncoder
# handle_unknown='ignore': Nếu gặp quận/huyện lạ lúc dự báo thì bỏ qua, không báo lỗi
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Gom lại thành bộ xử lý trung tâm
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Tạo Pipeline tổng: Xử lý -> Model RandomForest
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# ==========================================
# 4. TÌM KIẾM THAM SỐ TỐT NHẤT (GRID SEARCH)
# ==========================================
print("\nĐang huấn luyện và tìm kiếm mô hình tối ưu (có thể mất vài phút)...")

# Bộ tham số để thử nghiệm
param_grid = {
    # Số lượng cây: Thử 100 và 200 (Càng nhiều càng chính xác nhưng chậm)
    'model__n_estimators': [100],
    
    # Độ sâu tối đa: Giới hạn để tránh học vẹt (Overfitting)
    'model__max_depth': [12],
    
    # Số mẫu tối thiểu để tách nút
    'model__min_samples_split': [2]
}

# GridSearchCV: Tự động chạy thử tất cả các trường hợp trên
search = GridSearchCV(
    full_pipeline, 
    param_grid, 
    cv=3,  # Chia 3 phần để kiểm tra chéo
    scoring='neg_mean_absolute_error', # Ưu tiên sai số thấp nhất
    n_jobs=-1, # Dùng tất cả nhân CPU để chạy nhanh hơn
    verbose=1
)

search.fit(X_train, y_train)

# ==========================================
# 5. KẾT QUẢ VÀ LƯU MODEL
# ==========================================
best_model = search.best_estimator_

print("\n---------------- KẾT QUẢ ----------------")
print(f"Tham số tốt nhất: {search.best_params_}")

# Đánh giá trên tập Test
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Sai số trung bình (MAE): {mae:,.3f} tỷ") # Giả sử đơn vị là tỷ
print(f"Độ chính xác (R2 Score): {r2:.4f} (Càng gần 1 càng tốt)")

# Lưu model

joblib.dump(best_model, 'house_price_model.pkl' )
print(f"\n[OK] Đã lưu mô hình tốt nhất vào file: {'house_price_model.pkl'}")
print(f"Các cột mô hình đã học: {best_model.named_steps['model'].feature_names_in_ if hasattr(best_model.named_steps['model'], 'feature_names_in_') else 'Check pipeline'}")