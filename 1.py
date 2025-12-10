import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. KHAI BÁO DANH SÁCH ĐẶC TRƯNG CHUẨN
# ==========================================
REQUIRED_FEATURES = [
    'Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài', 'Rộng',
    'Quận_Huyện Gia Lâm', 'Quận_Huyện Hoài Đức', 'Quận_Huyện Mê Linh', 
    'Quận_Huyện Thanh Trì', 'Quận_Huyện Thạch Thất', 'Quận_Huyện Đông Anh', 
    'Quận_Quận Ba Đình', 'Quận_Quận Bắc Từ Liêm', 'Quận_Quận Cầu Giấy', 
    'Quận_Quận Hai Bà Trưng', 'Quận_Quận Hoàn Kiếm', 'Quận_Quận Hoàng Mai', 
    'Quận_Quận Hà Đông', 'Quận_Quận Long Biên', 'Quận_Quận Nam Từ Liêm', 
    'Quận_Quận Thanh Xuân', 'Quận_Quận Tây Hồ', 'Quận_Quận Đống Đa', 
    'Quận_Thị xã Sơn Tây', 'Huyện_Phường Bùi Thị Xuân', 'Huyện_Phường Bưởi', 
    'Huyện_Phường Bạch Mai', 'Huyện_Phường Bạch Đằng', 'Huyện_Phường Bồ Đề', 
    'Huyện_Phường Chương Dương', 'Huyện_Phường Cát Linh', 'Huyện_Phường Cầu Diễn', 
    'Huyện_Phường Cầu Dền', 'Huyện_Phường Cống Vị', 'Huyện_Phường Cổ Nhuế 1', 
    'Huyện_Phường Cổ Nhuế 2', 'Huyện_Phường Cửa Nam', 'Huyện_Phường Cửa Đông', 
    'Huyện_Phường Cự Khối', 'Huyện_Phường Dương Nội', 'Huyện_Phường Dịch Vọng', 
    'Huyện_Phường Dịch Vọng Hậu', 'Huyện_Phường Gia Thụy', 'Huyện_Phường Giang Biên', 
    'Huyện_Phường Giáp Bát', 'Huyện_Phường Giảng Võ', 'Huyện_Phường Hoàng Liệt', 
    'Huyện_Phường Hoàng Văn Thụ', 'Huyện_Phường Hà Cầu', 'Huyện_Phường Hàng Buồm', 
    'Huyện_Phường Hàng Bài', 'Huyện_Phường Hàng Bông', 'Huyện_Phường Hàng Bồ', 
    'Huyện_Phường Hàng Bột', 'Huyện_Phường Hàng Gai', 'Huyện_Phường Hàng Mã', 
    'Huyện_Phường Hàng Đào', 'Huyện_Phường Hạ Đình', 'Huyện_Phường Khâm Thiên', 
    'Huyện_Phường Khương Mai', 'Huyện_Phường Khương Thượng', 'Huyện_Phường Khương Trung', 
    'Huyện_Phường Khương Đình', 'Huyện_Phường Kim Giang', 'Huyện_Phường Kim Liên', 
    'Huyện_Phường Kim Mã', 'Huyện_Phường Kiến Hưng', 'Huyện_Phường La Khê', 
    'Huyện_Phường Liên Mạc', 'Huyện_Phường Liễu Giai', 'Huyện_Phường Long Biên', 
    'Huyện_Phường Láng Hạ', 'Huyện_Phường Láng Thượng', 'Huyện_Phường Lê Đại Hành', 
    'Huyện_Phường Lý Thái Tổ', 'Huyện_Phường Lĩnh Nam', 'Huyện_Phường Mai Dịch', 
    'Huyện_Phường Mai Động', 'Huyện_Phường Minh Khai', 'Huyện_Phường Mễ Trì', 
    'Huyện_Phường Mộ Lao', 'Huyện_Phường Mỹ Đình 1', 'Huyện_Phường Mỹ Đình 2', 
    'Huyện_Phường Nam Đồng', 'Huyện_Phường Nghĩa Tân', 'Huyện_Phường Nghĩa Đô', 
    'Huyện_Phường Nguyễn Du', 'Huyện_Phường Nguyễn Trung Trực', 'Huyện_Phường Nguyễn Trãi', 
    'Huyện_Phường Ngã Tư Sở', 'Huyện_Phường Ngô Thì Nhậm', 'Huyện_Phường Ngọc Hà', 
    'Huyện_Phường Ngọc Khánh', 'Huyện_Phường Ngọc Lâm', 'Huyện_Phường Ngọc Thụy', 
    'Huyện_Phường Nhân Chính', 'Huyện_Phường Nhật Tân', 'Huyện_Phường Phú Diễn', 
    'Huyện_Phường Phú La', 'Huyện_Phường Phú Lãm', 'Huyện_Phường Phú Lương', 
    'Huyện_Phường Phú Thượng', 'Huyện_Phường Phú Đô', 'Huyện_Phường Phúc Diễn', 
    'Huyện_Phường Phúc La', 'Huyện_Phường Phúc Lợi', 'Huyện_Phường Phúc Tân', 
    'Huyện_Phường Phúc Xá', 'Huyện_Phường Phúc Đồng', 'Huyện_Phường Phương Canh', 
    'Huyện_Phường Phương Liên', 'Huyện_Phường Phương Liệt', 'Huyện_Phường Phương Mai', 
    'Huyện_Phường Phạm Đình Hổ', 'Huyện_Phường Phố Huế', 'Huyện_Phường Quan Hoa', 
    'Huyện_Phường Quang Trung', 'Huyện_Phường Quán Thánh', 'Huyện_Phường Quảng An', 
    'Huyện_Phường Quốc Tử Giám', 'Huyện_Phường Quỳnh Lôi', 'Huyện_Phường Quỳnh Mai', 
    'Huyện_Phường Sài Đồng', 'Huyện_Phường Thanh Lương', 'Huyện_Phường Thanh Nhàn', 
    'Huyện_Phường Thanh Trì', 'Huyện_Phường Thanh Xuân Bắc', 'Huyện_Phường Thanh Xuân Nam', 
    'Huyện_Phường Thanh Xuân Trung', 'Huyện_Phường Thành Công', 'Huyện_Phường Thượng Thanh', 
    'Huyện_Phường Thượng Đình', 'Huyện_Phường Thạch Bàn', 'Huyện_Phường Thịnh Liệt', 
    'Huyện_Phường Thịnh Quang', 'Huyện_Phường Thổ Quan', 'Huyện_Phường Thụy Khuê', 
    'Huyện_Phường Thụy Phương', 'Huyện_Phường Trung Hoà', 'Huyện_Phường Trung Liệt', 
    'Huyện_Phường Trung Phụng', 'Huyện_Phường Trung Tự', 'Huyện_Phường Trung Văn', 
    'Huyện_Phường Trúc Bạch', 'Huyện_Phường Trương Định', 'Huyện_Phường Trần Hưng Đạo', 
    'Huyện_Phường Trần Phú', 'Huyện_Phường Tân Mai', 'Huyện_Phường Tây Mỗ', 
    'Huyện_Phường Tây Tựu', 'Huyện_Phường Tương Mai', 'Huyện_Phường Tứ Liên', 
    'Huyện_Phường Việt Hưng', 'Huyện_Phường Văn Chương', 'Huyện_Phường Văn Miếu', 
    'Huyện_Phường Văn Quán', 'Huyện_Phường Vĩnh Hưng', 'Huyện_Phường Vĩnh Phúc', 
    'Huyện_Phường Vĩnh Tuy', 'Huyện_Phường Vạn Phúc', 'Huyện_Phường Xuân La', 
    'Huyện_Phường Xuân Phương', 'Huyện_Phường Xuân Tảo', 'Huyện_Phường Xuân Đỉnh', 
    'Huyện_Phường Yên Hoà', 'Huyện_Phường Yên Nghĩa', 'Huyện_Phường Yên Phụ', 
    'Huyện_Phường Yên Sở', 'Huyện_Phường Yết Kiêu', 'Huyện_Phường Ô Chợ Dừa', 
    'Huyện_Phường Điện Biên', 'Huyện_Phường Đông Ngạc', 'Huyện_Phường Đại Kim', 
    'Huyện_Phường Đại Mỗ', 'Huyện_Phường Định Công', 'Huyện_Phường Đống Mác', 
    'Huyện_Phường Đồng Mai', 'Huyện_Phường Đồng Nhân', 'Huyện_Phường Đồng Tâm', 
    'Huyện_Phường Đồng Xuân', 'Huyện_Phường Đội Cấn', 'Huyện_Phường Đức Giang', 
    'Huyện_Phường Đức Thắng', 'Huyện_Thị trấn Quang Minh', 'Huyện_Thị trấn Trâu Quỳ', 
    'Huyện_Thị trấn Trạm Trôi', 'Huyện_Thị trấn Văn Điển', 'Huyện_Thị trấn Yên Viên', 
    'Huyện_Unknown', 'Huyện_Xã Bát Tràng', 'Huyện_Xã Bình Yên', 'Huyện_Xã Di Trạch', 
    'Huyện_Xã Hữu Hoà', 'Huyện_Xã Kim Chung', 'Huyện_Xã La Phù', 'Huyện_Xã Ngũ Hiệp', 
    'Huyện_Xã Ngọc Hồi', 'Huyện_Xã Sơn Đông', 'Huyện_Xã Tam Hiệp', 'Huyện_Xã Thanh Liệt', 
    'Huyện_Xã Thạch Hoà', 'Huyện_Xã Tân Triều', 'Huyện_Xã Tả Thanh Oai', 
    'Huyện_Xã Tứ Hiệp', 'Huyện_Xã Uy Nỗ', 'Huyện_Xã Vân Canh', 'Huyện_Xã Vân Hòa', 
    'Huyện_Xã Vĩnh Quỳnh', 'Huyện_Xã Đa Tốn', 'Huyện_Xã Đông Dư', 'Huyện_Xã Đông La', 
    'Loại hình nhà ở_Nhà mặt phố, mặt tiền', 'Loại hình nhà ở_Nhà ngõ, hẻm', 
    'Loại hình nhà ở_Nhà phố liền kề', 'Loại hình nhà ở_Unknown', 
    'Giấy tờ pháp lý_Unknown', 'Giấy tờ pháp lý_Đang chờ sổ', 'Giấy tờ pháp lý_Đã có sổ'
]

# ==========================================
# 2. LOAD VÀ XỬ LÝ DỮ LIỆU
# ==========================================
print("Đang đọc dữ liệu...")
try:
    # Đọc file CSV
    df = pd.read_csv('processed_housing_data.zip')
    print("Dữ liệu đầu vào:", df.shape)
    
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'processed_housing_data.csv'.")
    exit()

# Lấy Target (Giá nhà)
if 'Giá nhà' not in df.columns:
    print("Lỗi: File dữ liệu thiếu cột 'Giá nhà'")
    print("Các cột hiện có:", list(df.columns))
    exit()

y = df['Giá nhà']

# ==========================================
# 3. ĐỒNG BỘ CỘT (QUAN TRỌNG NHẤT)
# ==========================================
print("Đang đồng bộ hóa dữ liệu theo danh sách chuẩn...")

# LƯU Ý QUAN TRỌNG:
# Vì file 'processed_housing_data.csv' đã được xử lý (đã có các cột Quận_Huyện...),
# ta KHÔNG chạy pd.get_dummies nữa mà dùng thẳng reindex để lọc cột.

# Lệnh reindex sẽ:
# 1. Chỉ giữ lại những cột có trong REQUIRED_FEATURES
# 2. Điền số 0 vào những cột thiếu (fill_value=0)
# 3. Loại bỏ những cột thừa không cần thiết
X = df.reindex(columns=REQUIRED_FEATURES, fill_value=0)

# Điền dữ liệu thiếu cho các cột số (nếu có) bằng 0
X = X.fillna(0)

print(f"Số lượng đặc trưng sau khi xử lý: {X.shape[1]}")
print(f"Số lượng mẫu: {X.shape[0]}")

# ==========================================
# 4. HUẤN LUYỆN MODEL
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nĐang huấn luyện mô hình (giới hạn độ sâu để file nhẹ)...")
# max_depth=12 giúp file model nhỏ gọn (<100MB) để up lên Github
model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=12, 
    min_samples_split=5,
    random_state=42, 
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 5. ĐÁNH GIÁ VÀ LƯU
# ==========================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n---------------- KẾT QUẢ ----------------")
print(f"Sai số trung bình (MAE): {mae:,.3f} tỷ")
print(f"Độ chính xác (R2 Score): {r2:.4f}")

# Lưu Model (có nén)
joblib.dump(model, 'house_price_model.pkl')

# Lưu danh sách cột để App sử dụng
joblib.dump(REQUIRED_FEATURES, 'model_columns.pkl')

print("\n[OK] Đã lưu 2 file:")
print("1. house_price_model.pkl (Model chính)")
print("2. model_columns.pkl (Danh sách tên cột cần thiết)")