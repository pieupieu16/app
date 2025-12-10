import pandas as pd
import numpy as np

# 1. TẢI DỮ LIỆU
df = pd.read_csv('VN_housing_dataset.csv')

# Xóa cột index thừa nếu có
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. HÀM LÀM SẠCH DỮ LIỆU CƠ BẢN
def clean_numeric_col(df, col, remove_str):
    if col in df.columns:
        # Xóa đơn vị, khoảng trắng và chuyển sang số
        df[col] = df[col].astype(str).str.replace(remove_str, '', regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Áp dụng làm sạch cho các cột
df = clean_numeric_col(df, 'Diện tích', ' m²')
df = clean_numeric_col(df, 'Số phòng ngủ', ' phòng')
df = clean_numeric_col(df, 'Dài', ' m')
df = clean_numeric_col(df, 'Rộng', ' m')
df['Số tầng'] = pd.to_numeric(df['Số tầng'], errors='coerce') # Ép kiểu số tầng

# Làm sạch cột Giá/m2 đặc biệt (xử lý dấu phẩy)
if df['Giá/m2'].dtype == 'object':
    df['Giá/m2'] = df['Giá/m2'].str.replace(' triệu/m²', '', regex=False)
    df['Giá/m2'] = df['Giá/m2'].str.replace(',', '.', regex=False)
    df['Giá/m2'] = pd.to_numeric(df['Giá/m2'], errors='coerce')

# 3. TẠO BIẾN MỤC TIÊU (TARGET VARIABLE)
df['Giá nhà'] = df['Diện tích'] * df['Giá/m2']
# Quan trọng: Xóa các dòng không tính được giá nhà (vì không thể train model nếu không có đáp án)
df = df.dropna(subset=['Giá nhà'])

# Xóa cột 'Giá/m2' để tránh data leakage (rò rỉ dữ liệu) khi train
df = df.drop(columns=['Giá/m2'])

# 4. FEATURE ENGINEERING (TẠO ĐẶC TRƯNG MỚI)
# Xử lý ngày tháng
df['Ngày'] = pd.to_datetime(df['Ngày'], errors='coerce')
df['Năm'] = df['Ngày'].dt.year
df['Tháng'] = df['Ngày'].dt.month
df = df.drop(columns=['Ngày'])

# 5. XỬ LÝ GIÁ TRỊ THIẾU (IMPUTATION)
# Với biến số: Điền bằng trung vị (median)
numeric_cols = ['Diện tích', 'Số tầng', 'Số phòng ngủ', 'Dài', 'Rộng']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Với biến phân loại: Điền bằng 'Unknown'
categorical_cols = ['Quận', 'Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# 6. MÃ HÓA (ENCODING) & CHUẨN BỊ DỮ LIỆU CUỐI CÙNG
# Xóa cột Địa chỉ vì quá chi tiết, khó dùng cho model đơn giản
df_model = df.drop(columns=['Địa chỉ'])
df=df[df['Giá nhà']>1000]
df['Giá nhà'] = df['Giá nhà'] / 1000
# One-Hot Encoding cho các biến phân loại
df_final = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# 7. KIỂM TRA KẾT QUẢ
print("Kích thước dữ liệu sau xử lý:", df_final.shape)
print("Các cột dữ liệu (5 dòng đầu):")
print(df_final.head())


# Bạn có thể lưu lại file đã xử lý nếu cần
df.to_csv('processed_housing_data.csv', index=False)