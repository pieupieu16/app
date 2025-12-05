import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json

# 1. Cấu hình trang web
st.set_page_config(page_title="Bản đồ Quy hoạch Demo", layout="wide")

st.title("🗺️ Bản đồ Quy hoạch & Giá đất Hà Nội")
st.write("Đây là bản đồ tương tác được xây dựng bằng Python + Folium.")

# 2. Tạo bản đồ nền (Lớp đáy)
# location=[21.0285, 105.8542]: Tọa độ tâm Hà Nội (Hồ Gươm)
# zoom_start=11: Độ phóng to ban đầu
m = folium.Map(location=[21.0285, 105.8542], zoom_start=11)

# 3. Thêm lớp dữ liệu (Lớp phủ)
# Ở đây tôi dùng file GeoJSON ranh giới các tỉnh thành/quận huyện (ví dụ minh họa)
# Trong thực tế, bạn sẽ thay đường link này bằng file quy hoạch đất của bạn.
geojson_url = "https://raw.githubusercontent.com/VIG-Open-Tech/vietnam-boundaries/main/hanoi_districts.geojson"

try:
    # Tải dữ liệu từ internet
    response = requests.get(geojson_url)
    hanoi_data = response.json()

    # Tạo lớp phủ màu sắc lên bản đồ
    folium.GeoJson(
        hanoi_data,
        name="Ranh giới Quận",
        style_function=lambda feature: {
            'fillColor': '#ffaf00', # Màu nền bên trong (Màu cam)
            'color': 'black',       # Màu viền (Màu đen)
            'weight': 2,            # Độ dày viền
            'fillOpacity': 0.3,     # Độ trong suốt (0.3 là mờ mờ để nhìn thấy đường phố bên dưới)
        },
        # Tạo popup: Khi bấm vào khu vực nào sẽ hiện tên khu vực đó
        tooltip=folium.GeoJsonTooltip(fields=['Name'], aliases=['Quận/Huyện:'])
    ).add_to(m)
    
    st.success("Đã tải xong lớp dữ liệu hành chính!")

except Exception as e:
    st.error(f"Không thể tải dữ liệu bản đồ: {e}")
    # Nếu lỗi, bản đồ vẫn hiện nhưng không có lớp phủ

# 4. Hiển thị bản đồ lên Streamlit
# width=100% để bản đồ rộng theo màn hình
st_folium(m, width=1200, height=600)