import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from datetime import datetime

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hệ thống Quản lý & Định giá BĐS Hà Nội",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .stMetric {background-color: white; padding: 10px; border-radius: 8px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM XỬ LÝ DỮ LIỆU (CORE) ---
@st.cache_data
def load_data(file_path='processed_housing_data.zip'):
    try:
        df = pd.read_csv(file_path)
        
        # 1. Xử lý tên cột
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        df = df.loc[:, ~df.columns.duplicated()]

        # 2. Hàm tái tạo cột phân loại từ One-Hot Encoding
        def reverse_ohe(row, prefix):
            cols = [c for c in df.columns if c.startswith(prefix)]
            for c in cols:
                if row[c] == 1:
                    return c.replace(prefix, '')
            return 'Khác'

        # 3. Tạo cột phân loại để hiển thị (Visual)
        if 'Quận' not in df.columns:
            df['Quận'] = df.apply(lambda x: reverse_ohe(x, 'Quận_'), axis=1)
        
        if 'Loại nhà' not in df.columns:
            df['Loại nhà'] = df.apply(lambda x: reverse_ohe(x, 'Loại hình nhà ở_'), axis=1)

        return df
    except Exception as e:
        # Không hiển thị lỗi ngay lúc đầu nếu chưa có file, trả về DataFrame rỗng
        return pd.DataFrame()

# --- 3. QUẢN LÝ STATE ---
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# --- 4. MENU ĐIỀU HƯỚNG ---
col_logo, col_text = st.columns([1, 5])
with col_text:
    st.title("Hệ thống Định giá BĐS Hà Nội")

selected = option_menu(
    menu_title=None,
    options=["Trang chủ", "Quản lý Dữ liệu (CRUD)", "Phân tích Trực quan", "Bảng điều khiển Tableau& Bản đồ quy hoạch Hà Nội"],
    icons=["house", "table", "bar-chart-line", "magic"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#a13d3d"},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
)

# Xác định tên cột dữ liệu chính
COL_PRICE = 'Giá nhà'
COL_AREA = 'Diện tích'
COL_DISTRICT = 'Quận'
COL_TYPE = 'Loại nhà'

# =========================================================
# MODULE 1: TRANG CHỦ & TABLEAU
# =========================================================
if selected == "Trang chủ":
    st.title(" Dashboard Tổng quan")
    
    # CSS Custom cho Metric
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] { font-size: 24px; }
        </style>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    
    if not df.empty and COL_PRICE in df.columns:
        num_houses = len(df)
        avg_price = df[COL_PRICE].mean()
        max_price = df[COL_PRICE].max()
        
        # Định nghĩa các tiền tố của cột Quận/Huyện (dựa trên dữ liệu bạn gửi)
        DISTRICT_PREFIXES_LIST = ['Quận_Huyện', 'Quận_Quận', 'Quận_Thị xã']
        # Định nghĩa các tiền tố cần loại bỏ để lấy tên Quận/Huyện
        PREFIXES_TO_REMOVE = ['Quận_Huyện ', 'Quận_Quận ', 'Quận_Thị xã ', 'Quận_'] 

        # Khởi tạo giá trị mặc định
        cheapest_district = "N/A"

        if COL_AREA in df.columns and COL_PRICE in df.columns:
            
            # 1. TẠO CỘT DISTRICT GỐC (DE-ONE-HOT ENCODING)
            try:
                # Lấy danh sách tất cả các cột Quận/Huyện One-Hot
                district_cols = [col for col in df.columns if any(col.startswith(p) for p in DISTRICT_PREFIXES_LIST)]
                
                if not district_cols:
                    cheapest_district = "Lỗi: Không tìm thấy cột Quận/Huyện (One-Hot)"
                else:
                    # Hàm để tái tạo lại tên Quận/Huyện
                    def get_district_name(row, cols, prefixes_to_remove):
                        # Tìm tên cột có giá trị lớn nhất (giá trị 1)
                        selected_col = row[cols].idxmax()
                        
                        # Kiểm tra để đảm bảo đó là 1, nếu không là 'Unknown'
                        if row[selected_col] == 1:
                            name = selected_col
                            for prefix in prefixes_to_remove:
                                if name.startswith(prefix):
                                    name = name[len(prefix):]
                                    break
                            return name
                        return 'Unknown' 

                    # Áp dụng hàm để tạo cột tên Quận/Huyện mới tạm thời
                    df['District_Name'] = df.apply(lambda row: get_district_name(row, district_cols, PREFIXES_TO_REMOVE), axis=1)

                    # 2. LỌC VÀ TÍNH TOÁN
                    
                    # Lọc dữ liệu hợp lệ: Diện tích > 0, Giá nhà > 0, và tên Quận/Huyện đã được xác định
                    valid_area = df[
                        (df[COL_AREA] > 0) & 
                        (df[COL_PRICE] > 0) &
                        (df['District_Name'] != 'Unknown')
                    ].copy()
                    
                    # Kiểm tra: Đảm bảo có đủ Quận/Huyện để so sánh
                    if valid_area['District_Name'].nunique() > 1:
                        valid_area['Price_per_m2'] = valid_area[COL_PRICE] / valid_area[COL_AREA]
                        
                        # Tính giá trung bình trên mỗi mét vuông theo Quận/Huyện
                        grouped_prices = valid_area.groupby('District_Name')['Price_per_m2'].mean()
                        
                        if not grouped_prices.empty:
                            cheapest_district = grouped_prices.idxmin()
                        else:
                            cheapest_district = "N/A (Không tính được giá trung bình)"
                    else:
                        cheapest_district = "N/A (Chỉ có 1 khu vực hoặc không đủ dữ liệu)"

            except Exception as e:
                cheapest_district = f"Lỗi xử lý dữ liệu: {str(e)}"\
                
        c1.metric("Số nhà đang bán", f"{num_houses:,}")
        c2.metric("Giá trung bình", f"{avg_price/1000:,.2f} Tỷ")
        c3.metric("Khu vực rẻ nhất (m²)", f"{cheapest_district}")
        c4.metric("Căn đắt nhất", f"{max_price/1000:,.2f} Tỷ")
    else:
        st.info("Vui lòng Import dữ liệu ở tab 'Quản lý Dữ liệu' để xem thống kê.")

    # 1. LOAD MODEL
  
    @st.cache_resource
    def load_model_assets():
        try:
            model = joblib.load('house_price_model.pkl')
            cols = joblib.load('model_columns.pkl')
            return model, cols
        except Exception as e:
            st.error(f"Không tìm thấy file model: {e}. Vui lòng kiểm tra lại thư mục.")
            return None, None

    model, model_columns = load_model_assets()

    if model is None:
        st.stop()

    # 2. DANH SÁCH DỮ LIỆU
    districts = [
        "Ba Đình", "Bắc Từ Liêm", "Cầu Giấy", "Đống Đa", "Hà Đông", "Hai Bà Trưng", 
        "Hoàn Kiếm", "Hoàng Mai", "Long Biên", "Nam Từ Liêm", "Tây Hồ", "Thanh Xuân",
        "Chương Mỹ", "Đan Phượng", "Đông Anh", "Gia Lâm", "Hoài Đức", "Mê Linh", 
        "Mỹ Đức", "Phú Xuyên", "Phúc Thọ", "Quốc Oai", "Sóc Sơn", "Thạch Thất", 
        "Thanh Oai", "Thanh Trì", "Thường Tín", "Thị xã Sơn Tây"
    ]
    districts.sort()

    # (Dữ liệu Wards map của bạn - giữ nguyên nhưng thu gọn hiển thị ở đây)
    wards_map = {
        # CÁC QUẬN NỘI THÀNH VÀ LÂN CẬN
        "Ba Đình": ["Phường Cống Vị", "Phường Giảng Võ", "Phường Kim Mã", "Phường Liễu Giai", 
                    "Phường Ngọc Hà", "Phường Ngọc Khánh", "Phường Phúc Xá", "Phường Quán Thánh", 
                    "Phường Thành Công", "Phường Trúc Bạch", "Phường Vĩnh Phúc", "Phường Đội Cấn", "Phường Điện Biên"],
        
        "Hoàn Kiếm": ["Phường Chương Dương", "Phường Cửa Nam", "Phường Cửa Đông", "Phường Hàng Buồm", 
                    "Phường Hàng Bài", "Phường Hàng Bông", "Phường Hàng Bạc", "Phường Hàng Bồ", 
                    "Phường Hàng Gai", "Phường Hàng Mã", "Phường Hàng Trống", "Phường Hàng Đào", 
                    "Phường Lý Thái Tổ", "Phường Phan Chu Trinh", "Phường Phúc Tân", "Phường Tràng Tiền", "Phường Đồng Xuân", "Phường Yết Kiêu"],
        
        "Hai Bà Trưng": ["Phường Bách Khoa", "Phường Bùi Thị Xuân", "Phường Bạch Mai", "Phường Bạch Đằng", 
                        "Phường Cầu Dền", "Phường Đồng Nhân", "Phường Đồng Tâm", "Phường Kim Liên", 
                        "Phường Lê Đại Hành", "Phường Minh Khai", "Phường Nguyễn Du", "Phường Ngô Thì Nhậm", 
                        "Phường Phạm Đình Hổ", "Phường Phố Huế", "Phường Quỳnh Lôi", "Phường Quỳnh Mai", 
                        "Phường Thanh Lương", "Phường Thanh Nhàn", "Phường Trương Định", "Phường Vĩnh Tuy", "Phường Đống Mác"],
        
        "Đống Đa": ["Phường Hàng Bột", "Phường Khâm Thiên", "Phường Khương Thượng", "Phường Kim Liên", 
                    "Phường Láng Hạ", "Phường Láng Thượng", "Phường Nam Đồng", "Phường Nguyễn Trãi", 
                    "Phường Ngã Tư Sở", "Phường Phương Liên", "Phường Phương Mai", "Phường Quốc Tử Giám", 
                    "Phường Thịnh Quang", "Phường Thổ Quan", "Phường Trung Liệt", "Phường Trung Phụng", 
                    "Phường Trung Tự", "Phường Văn Chương", "Phường Văn Miếu", "Phường Ô Chợ Dừa"],
        
        "Cầu Giấy": ["Phường Cầu Diễn", "Phường Dịch Vọng", "Phường Dịch Vọng Hậu", "Phường Mai Dịch", 
                    "Phường Nghĩa Tân", "Phường Nghĩa Đô", "Phường Quan Hoa", "Phường Trung Hoà", "Phường Yên Hoà"],
        
        "Tây Hồ": ["Phường Bưởi", "Phường Nhật Tân", "Phường Quảng An", "Phường Thụy Khuê", 
                "Phường Tứ Liên", "Phường Xuân La", "Phường Yên Phụ"],

        "Thanh Xuân": ["Phường Hạ Đình", "Phường Khương Mai", "Phường Khương Trung", "Phường Khương Đình", 
                    "Phường Kim Giang", "Phường Nhân Chính", "Phường Phương Liệt", "Phường Thanh Xuân Bắc", 
                    "Phường Thanh Xuân Nam", "Phường Thanh Xuân Trung", "Phường Thượng Đình", "Phường Định Công"], # Định Công thường thuộc Hoàng Mai nhưng có thể liên quan
        
        "Hoàng Mai": ["Phường Giáp Bát", "Phường Hoàng Liệt", "Phường Hoàng Văn Thụ", "Phường Lĩnh Nam", 
                    "Phường Mai Động", "Phường Thịnh Liệt", "Phường Trần Phú", "Phường Tân Mai", 
                    "Phường Tương Mai", "Phường Vĩnh Hưng", "Phường Yên Sở", "Phường Đại Kim", 
                    "Phường Định Công", "Phường Đồng Tâm", "Phường Vĩnh Tuy", "Phường Thanh Trì"], # (Phường, không phải Huyện)
        
        "Long Biên": ["Phường Bồ Đề", "Phường Cự Khối", "Phường Gia Thụy", "Phường Giang Biên", 
                    "Phường Long Biên", "Phường Ngọc Lâm", "Phường Ngọc Thụy", "Phường Phúc Đồng", 
                    "Phường Phúc Lợi", "Phường Phúc Tân", "Phường Phúc Xá", "Phường Sài Đồng", 
                    "Phường Thạch Bàn", "Phường Thượng Thanh", "Phường Việt Hưng", "Phường Đức Giang"],

        "Bắc Từ Liêm": ["Phường Cầu Diễn", "Phường Cổ Nhuế 1", "Phường Cổ Nhuế 2", "Phường Liên Mạc", 
                        "Phường Minh Khai", "Phường Phú Diễn", "Phường Phúc Diễn", "Phường Thượng Cát", 
                        "Phường Thụy Phương", "Phường Tây Tựu", "Phường Xuân Tảo", "Phường Xuân Đỉnh", "Phường Đông Ngạc", "Phường Đức Thắng"],
        
        "Nam Từ Liêm": ["Phường Cầu Diễn", "Phường Mễ Trì", "Phường Mỹ Đình 1", "Phường Mỹ Đình 2", 
                        "Phường Phú Đô", "Phường Phương Canh", "Phường Trung Văn", "Phường Tây Mỗ", "Phường Xuân Phương", "Phường Đại Mỗ"],
        
        "Hà Đông": ["Phường Dương Nội", "Phường Hà Cầu", "Phường Kiến Hưng", "Phường La Khê", 
                    "Phường Mộ Lao", "Phường Nguyễn Trãi", "Phường Phú La", "Phường Phú Lãm", 
                    "Phường Phú Lương", "Phường Phú Thịnh", "Phường Phúc La", "Phường Quang Trung", 
                    "Phường Vạn Phúc", "Phường Văn Quán", "Phường Yên Nghĩa", "Phường Đồng Mai"],
        
        # CÁC HUYỆN VÀ THỊ XÃ
        "Đông Anh": ["Thị trấn Đông Anh", "Xã Bắc Hồng", "Xã Dục Tú", "Xã Hải Bối", "Xã Kim Chung", 
                    "Xã Kim Nỗ", "Xã Liên Hà", "Xã Mai Lâm", "Xã Nam Hồng", "Xã Nguyên Khê", 
                    "Xã Tiên Dương", "Xã Uy Nỗ", "Xã Vân Nội", "Xã Võng La", "Xã Xuân Giang", 
                    "Xã Xuân Nộn", "Xã Yên Thường", "Xã Đại Mạch", "Xã Đông Hội"],
        
        "Gia Lâm": ["Thị trấn Trâu Quỳ", "Thị trấn Yên Viên", "Xã Bát Tràng", "Xã Cổ Bi", "Xã Cự Khối", 
                    "Xã Đa Tốn", "Xã Kiêu Kỵ", "Xã Ninh Hiệp", "Xã Phú Thị", "Xã Phù Đổng", 
                    "Xã Trung Mầu", "Xã Yên Viên", "Xã Đông Dư", "Xã Đặng Xá", "Xã Đình Xuyên"],
        
        "Hoài Đức": ["Thị trấn Trạm Trôi", "Xã An Khánh", "Xã An Thượng", "Xã Cát Quế", "Xã Di Trạch", 
                    "Xã Dương Liễu", "Xã Lại Yên", "Xã La Phù", "Xã Song Phương", "Xã Sơn Đồng", 
                    "Xã Tiền Yên", "Xã Vân Canh", "Xã Vân Côn", "Xã Yên Sở", "Xã Đông La", "Xã Đức Thượng"],
        
        "Thanh Trì": ["Thị trấn Văn Điển", "Xã Duyên Hà", "Xã Duyên Thái", "Xã Hữu Hoà", "Xã Khánh Hà", 
                    "Xã Liên Ninh", "Xã Ngọc Hồi", "Xã Ngũ Hiệp", "Xã Tả Thanh Oai", "Xã Tam Hiệp", 
                    "Xã Tân Triều", "Xã Tứ Hiệp", "Xã Vĩnh Quỳnh", "Xã Văn Bình", "Xã Yên Mỹ", 
                    "Xã Thanh Liệt"], # (Loại trừ các phường đã xếp vào Quận khác)
        
        "Thạch Thất": ["Thị trấn Liên Quan", "Xã Bình Phú", "Xã Bình Yên", "Xã Cẩm Quan", "Xã Cổ Đông", 
                    "Xã Hạ Bằng", "Xã Hữu Bằng", "Xã Hương Ngải", "Xã Kim Quan", "Xã Lại Thượng", 
                    "Xã Phú Kim", "Xã Phú Mãn", "Xã Phùng Xá", "Xã Tân Xã", "Xã Thạch Hoà", 
                    "Xã Tiên Xuân", "Xã Yên Bình", "Xã Yên Trung", "Xã Canh Nậu", "Xã Đồng Trúc"],
        
        "Sóc Sơn": ["Thị trấn Sóc Sơn", "Xã Bắc Sơn", "Xã Hiền Ninh", "Xã Kim Lũ", "Xã Mai Đình", 
                    "Xã Minh Phú", "Xã Minh Trí", "Xã Nam Sơn", "Xã Phù Linh", "Xã Phù Lỗ", 
                    "Xã Quang Tiến", "Xã Tân Dân", "Xã Thanh Xuân", "Xã Tiên Dược", "Xã Trung Giã", 
                    "Xã Việt Long", "Xã Xuân Giang", "Xã Xuân Thu"], # (Loại trừ các phường/xã đã xếp vào Quận khác)

        "Thường Tín": ["Thị trấn Thường Tín", "Xã Hà Hồi", "Xã Hiền Giang", "Xã Hòa Bình", "Xã Hồng Vân", 
                    "Xã Khánh Hà", "Xã Lê Lợi", "Xã Liên Phương", "Xã Minh Cường", "Xã Nghiêm Xuyên", 
                    "Xã Nhị Khê", "Xã Ninh Sở", "Xã Quất Động", "Xã Thắng Lợi", "Xã Thống Nhất", 
                    "Xã Tiền Phong", "Xã Tô Hiệu", "Xã Tự Nhiên", "Xã Vạn Điểm", "Xã Văn Bình", "Xã Văn Phú"],

        "Chương Mỹ": ["Thị trấn Chúc Sơn", "Thị trấn Xuân Mai", "Xã Hợp Thanh", "Xã Nam Phương Tiến", "Xã Phụng Châu", 
                    "Xã Thủy Xuân Tiên", "Xã Đông Phương Yên", "Xã Trung Hòa", "Xã Văn Võ", "Xã Đồng Lạc"],
        
        "Đan Phượng": ["Thị trấn Phùng", "Xã Đan Phượng", "Xã Đồng Tháp", "Xã Hạ Mỗ", "Xã Hồng Hà", 
                    "Xã Liên Hà", "Xã Liên Hồng", "Xã Phương Đình", "Xã Song Phượng", "Xã Thọ An", 
                    "Xã Thọ Xuân", "Xã Thượng Mỗ", "Xã Trung Châu"],
        
        "Phú Xuyên": ["Thị trấn Phú Xuyên", "Xã Bạch Hạ", "Xã Châu Can", "Xã Chuyên Mỹ", "Xã Đại Thắng", 
                    "Xã Hồng Thái", "Xã Khai Thái", "Xã Minh Tân", "Xã Nam Phong", "Xã Nam Triều", 
                    "Xã Phú Châu", "Xã Phú Túc", "Xã Phúc Tiến", "Xã Quang Lãng", "Xã Quang Trung", 
                    "Xã Sơn Hà", "Xã Tân Dân", "Xã Tri Thủy", "Xã Tri Trung", "Xã Văn Hoàng", "Xã Vân Từ"],
        
        "Quốc Oai": ["Thị trấn Quốc Oai", "Xã Cấn Hữu", "Xã Cộng Hòa", "Xã Đại Thành", "Xã Đồng Quang", 
                    "Xã Hòa Thạch", "Xã Liệp Tuyết", "Xã Ngọc Liệp", "Xã Ngọc Mỹ", "Xã Phú Cát", 
                    "Xã Phú Mãn", "Xã Phượng Cách", "Xã Sài Sơn", "Xã Tuyết Nghĩa", "Xã Yên Sơn"],
        
        "Thị xã Sơn Tây": ["Phường Lê Lợi", "Phường Ngô Quyền", "Phường Phú Thịnh", "Phường Quang Trung", 
                        "Phường Sơn Lộc", "Phường Trung Hưng", "Phường Viên Sơn", "Phường Xuân Khanh", 
                        "Xã Cổ Đông", "Xã Đường Lâm", "Xã Kim Sơn", "Xã Sơn Đông", "Xã Thanh Mỹ", "Xã Xuân Sơn"],
        
        "Mê Linh": ["Thị trấn Quang Minh", "Xã Chu Phan", "Xã Đại Thịnh", "Xã Hoàng Kim", "Xã Kim Hoa", 
                    "Xã Liên Mạc", "Xã Mê Linh", "Xã Tam Đồng", "Xã Thạch Đà", "Xã Tiền Phong", 
                    "Xã Tráng Việt", "Xã Tự Lập", "Xã Văn Khê", "Xã Vạn Yên", "Xã Thanh Lâm"],
        
        "Phúc Thọ": ["Thị trấn Phúc Thọ", "Xã Cẩm Đình", "Xã Hát Môn", "Xã Hiệp Thuận", "Xã Liên Hiệp", 
                    "Xã Long Xuyên", "Xã Ngọc Tảo", "Xã Phụng Thượng", "Xã Sen Chiểu", "Xã Tam Thuấn", 
                    "Xã Thanh Đa", "Xã Thượng Cốc", "Xã Tích Giang", "Xã Vân Hà", "Xã Vân Nam", "Xã Võng Xuyên", "Xã Xuân Phú"],
        
        "Mỹ Đức": ["Thị trấn Đại Nghĩa", "Xã An Mỹ", "Xã An Phú", "Xã Bột Xuyên", "Xã Đại Hưng", 
                "Xã Đồng Tâm", "Xã Hồng Sơn", "Xã Hợp Thanh", "Xã Hợp Tiến", "Xã Hùng Tiến", 
                "Xã Hương Sơn", "Xã Lê Thanh", "Xã Mỹ Thành", "Xã Phù Lưu Tế", "Xã Phúc Lâm", 
                "Xã Thượng Lâm", "Xã Tuy Lai", "Xã Vạn Kim"],
        
        "Thanh Oai": ["Thị trấn Kim Bài", "Xã Bích Hòa", "Xã Cự Khê", "Xã Dân Hòa", "Xã Hồng Dương", 
                    "Xã Kim An", "Xã Kim Thư", "Xã Liên Châu", "Xã Mỹ Hưng", "Xã Phương Trung", 
                    "Xã Tam Hưng", "Xã Thanh Cao", "Xã Thanh Mai", "Xã Thanh Văn", "Xã Xuân Dương"],
        
        
    }

    # Extract Features Names from Model
    house_types = sorted([c.replace('Loại hình nhà ở_', '') for c in model_columns if c.startswith('Loại hình nhà ở_')])
    legal_types = sorted([c.replace('Giấy tờ pháp lý_', '') for c in model_columns if c.startswith('Giấy tờ pháp lý_')])

    # 3. GIAO DIỆN NHẬP LIỆU (KHÔNG DÙNG ST.FORM ĐỂ CÓ TƯƠNG TÁC TỨC THÌ)
    st.subheader("📋 Thông tin Bất động sản")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dien_tich = st.number_input("Diện tích (m²)", 10.0, 5000.0, 50.0)
        chieu_rong = st.number_input("Mặt tiền (m)", 1.0, 50.0, 5.0)
    with col2:
        chieu_dai = st.number_input("Chiều dài (m)", 1.0, 100.0, 10.0)
        so_tang = st.number_input("Số tầng", 1, 50, 3)
    with col3:
        so_phong = st.number_input("Số phòng ngủ", 1, 20, 2)
        nam_gd = st.number_input("Năm", 2000, 2030, datetime.now().year)
        thang_gd = st.number_input("Tháng", 1, 12, datetime.now().month)

    st.markdown("---")
    st.subheader("📍 Vị trí & Đặc điểm")
    
    c4, c5 = st.columns(2)
    with c4:
        # Tương tác: Chọn Quận -> Cập nhật danh sách Phường
        selected_district = st.selectbox("Quận / Huyện", districts)
        
        # Lấy danh sách phường tương ứng
        available_wards = wards_map.get(selected_district, [])
        
        # Checkbox Logic: Nếu check -> Enable Dropdown
        is_ward_specific = st.checkbox("Chọn Phường/Xã chi tiết?", value=False)
        
        selected_ward = st.selectbox(
            "Phường / Xã", 
            options=available_wards if available_wards else ["Chưa có dữ liệu"],
            disabled=not is_ward_specific  # Disable nếu KHÔNG check
        )
    
    with c5:
        selected_type = st.selectbox("Loại hình nhà", house_types)
        selected_legal = st.selectbox("Pháp lý", legal_types)

    # Nút Dự báo (Nằm ngoài cùng để gom logic)
    st.markdown("###")
    predict_btn = st.button("💰 DỰ BÁO GIÁ NHÀ", type="primary", use_container_width=True)

    # 4. XỬ LÝ DỰ BÁO
    if predict_btn:
        # A. Tạo DataFrame rỗng chuẩn theo Model
        input_data = pd.DataFrame(index=[0], columns=model_columns).fillna(0)

        # B. Điền dữ liệu số
        input_data['Diện tích'] = dien_tich
        input_data['Dài'] = chieu_dai
        input_data['Rộng'] = chieu_rong
        input_data['Số tầng'] = so_tang
        input_data['Số phòng ngủ'] = so_phong
        input_data['Năm'] = nam_gd
        input_data['Tháng'] = thang_gd

        # C. Điền dữ liệu One-Hot
        def set_one_hot(prefix, value):
            col = f"{prefix}{value}"
            if col in input_data.columns:
                input_data[col] = 1
        
        set_one_hot('Quận_', selected_district)
        set_one_hot('Loại hình nhà ở_', selected_type)
        set_one_hot('Giấy tờ pháp lý_', selected_legal)
        
        if is_ward_specific and selected_ward:
            set_one_hot('Huyện_', selected_ward) # Lưu ý: Model đang dùng prefix 'Huyện_' cho Phường/Xã?

        # D. Predict
        with st.spinner("Đang tính toán..."):
            try:
                predicted_price = model.predict(input_data)[0]
                
                # Xử lý giá trị âm/quá nhỏ
                if predicted_price <= 0:
                    predicted_price = 0.01

                st.success("Tính toán hoàn tất!")
                
                res_c1, res_c2 = st.columns([2, 1])
                with res_c1:
                    st.markdown(f"""
                    <div style="background-color: #f0fff4; padding: 20px; border-radius: 10px; border: 2px solid #48bb78; text-align: center;">
                        <h3 style="color: #2f855a; margin:0;">GIÁ TRỊ ƯỚC TÍNH</h3>
                        <h1 style="color: #22543d; font-size: 50px; margin: 10px 0;">{predicted_price:,.2f} Tỷ</h1>
                        <p style="color: #718096;">~ {(predicted_price * 1_000_000_000 / dien_tich):,.0f} VNĐ / m²</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with res_c2:
                    st.info("Chi tiết đầu vào")
                    st.write(f"**Vị trí:** {selected_district}")
                    if is_ward_specific:
                        st.write(f"**Phường:** {selected_ward}")
                    st.write(f"**Diện tích:** {dien_tich} m²")
                    st.write(f"**Kết cấu:** {so_tang} tầng, {so_phong} PN")

            except Exception as e:
                st.error(f"Lỗi khi dự báo: {str(e)}")
                st.dataframe(input_data) # Debug


   

# =========================================================
# MODULE 2: QUẢN LÝ DỮ LIỆU
# =========================================================
elif selected == "Quản lý Dữ liệu (CRUD)":
    st.title(" Quản lý Dữ liệu")
    
    with st.expander(" Import Dữ liệu mới (CSV)"):
        uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.session_state.df = new_df
                st.success("Import thành công!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi file: {e}")

    st.subheader(" Tìm kiếm & Lọc")
    col_search, col_filter = st.columns(2)
    with col_search:
        search_term = st.text_input("Tìm kiếm (Quận/Loại nhà):")
    with col_filter:
        price_range = st.slider("Khoảng giá (Triệu)", 0.0, 100.0, (0.0, df[COL_PRICE].max() ))
    
    filtered_df = df.copy()
    if not filtered_df.empty:
        if COL_PRICE in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df[COL_PRICE] >= price_range[0]) & (filtered_df[COL_PRICE] <= price_range[1])]
        
        if search_term:
            # Tạo mask tìm kiếm an toàn
            mask = pd.Series(False, index=filtered_df.index)
            if COL_DISTRICT in filtered_df.columns:
                mask |= filtered_df[COL_DISTRICT].str.contains(search_term, case=False, na=False)
            if COL_TYPE in filtered_df.columns:
                mask |= filtered_df[COL_TYPE].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]

        st.info(f"Hiển thị {len(filtered_df)} bản ghi.")
        edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True)

        if st.button(" Lưu thay đổi"):
            st.session_state.df = edited_df
            st.success("Đã lưu dữ liệu tạm thời (Reload trang sẽ mất nếu không lưu xuống file)!")
    else:
        st.warning("Chưa có dữ liệu.")

# =========================================================
# MODULE 3: PHÂN TÍCH TRỰC QUAN
# =========================================================
elif selected == "Phân tích Trực quan":
    st.title(" Phân tích Giá trị BĐS")

    if df.empty or COL_PRICE not in df.columns:
        st.warning("Chưa có dữ liệu hoặc cột 'Giá nhà' không tồn tại.")
        st.stop()

    tab1, tab2, tab3 = st.tabs([" Vị trí & Giá", " Đặc điểm & Giá", " Tương quan"])

    with tab1:
        st.subheader("Giá trung bình theo Quận")
        if COL_DISTRICT in df.columns:
            avg_price_quan = df.groupby(COL_DISTRICT)[COL_PRICE].mean().sort_values(ascending=False).reset_index()
            fig_bar = px.bar(avg_price_quan, x=COL_DISTRICT, y=COL_PRICE, color=COL_PRICE,
                             labels={COL_PRICE: 'Giá TB (Tỷ)'})
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Phân tích theo Loại hình")
        col_a, col_b = st.columns(2)
        with col_a:
            if COL_TYPE in df.columns:
                type_counts = df[COL_TYPE].value_counts().reset_index()
                type_counts.columns = [COL_TYPE, 'Số lượng']
                fig_pie = px.pie(type_counts, values='Số lượng', names=COL_TYPE, title="Tỷ lệ Loại hình")
                st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            if COL_AREA in df.columns:
                fig_scatter = px.scatter(df, x=COL_AREA, y=COL_PRICE, color=COL_TYPE if COL_TYPE in df.columns else None, 
                                         title="Diện tích vs Giá")
                st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("Ma trận tương quan")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        # Chọn các cột quan trọng từ danh sách mới
        potential_cols = [COL_PRICE, COL_AREA, 'Số phòng ngủ', 'Số tầng', 'Rộng', 'Dài']
        valid_cols = [c for c in potential_cols if c in numeric_df.columns]
        
        if valid_cols:
            corr_matrix = numeric_df[valid_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# MODULE 4: DỰ BÁO GIÁ (ĐÃ SỬA LỖI LOGIC FORM)
# =========================================================
elif selected == "Bảng điều khiển Tableau& Bản đồ quy hoạch Hà Nội":
    
    st.markdown("---") # Đường kẻ phân cách cho đẹp
    st.subheader(" Thông tin quy hoạch")

    # Thêm nút bấm liên kết
    st.link_button(
        label="🗺️ Mở Bản đồ quy hoạch Hà Nội", 
        url="https://qhkhsdd.hanoi.gov.vn/datdai?link=63d567a66c54",
        help="Nhấn để xem bản đồ quy hoạch chi tiết trên trang của Sở Tài nguyên và Môi trường Hà Nội"
    )

    st.divider()
    st.subheader(" Tableau Visualization")
    
    tableau_code = """
    <div class='tableauPlaceholder' id='viz1763483099173' style='position: relative'><noscript><a href='#'><img alt='tk ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Book7_17631271401140&#47;tk' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1763483099173');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                                            vizElement.parentNode.insertBefore(scriptElement, vizElement);                
                                            </script>
    """
    components.html(tableau_code, height=850, scrolling=True)

    