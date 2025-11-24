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
            # Prefix mới là 'Quận_' (Ví dụ: Quận_Quận Ba Đình)
            df['Quận'] = df.apply(lambda x: reverse_ohe(x, 'Quận_'), axis=1)
        
        if 'Loại nhà' not in df.columns:
            # Prefix mới là 'Loại hình nhà ở_'
            df['Loại nhà'] = df.apply(lambda x: reverse_ohe(x, 'Loại hình nhà ở_'), axis=1)

        # 4. Đảm bảo có cột 'Giá nhà' và 'Diện tích' cho biểu đồ
        # Nếu dữ liệu đã dùng tên tiếng Việt thì không cần rename, nhưng check cho chắc
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
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
    options=["Trang chủ & Tableau", "Quản lý Dữ liệu (CRUD)", "Phân tích Trực quan", "Dự báo Giá nhà"],
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

# Xác định tên cột dữ liệu chính (theo features mới)
COL_PRICE = 'Giá nhà'
COL_AREA = 'Diện tích'
COL_DISTRICT = 'Quận'
COL_TYPE = 'Loại nhà'

# =========================================================
# MODULE 1: TRANG CHỦ & TABLEAU
# =========================================================
if selected == "Trang chủ & Tableau":
    st.title(" Dashboard Tổng quan")
    
    # CSS Custom
    st.markdown("""
        <style>
        [data-testid="stMetric"] { background-color: #000000 !important; border: 1px solid #00ff00; }
        [data-testid="stMetricLabel"] p { color: #00ff00 !important; font-weight: bold; }
        [data-testid="stMetricValue"] div { color: #00ff00 !important; }
        </style>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    
    if not df.empty and COL_PRICE in df.columns:
        num_houses = len(df)
        avg_price = df[COL_PRICE].mean()
        max_price = df[COL_PRICE].max()
        
        # Tính khu vực rẻ nhất
        if COL_AREA in df.columns:
            valid_area = df[df[COL_AREA] > 0].copy()
            valid_area['Price_per_m2'] = valid_area[COL_PRICE] / valid_area[COL_AREA]
            cheapest_district = valid_area.groupby(COL_DISTRICT)['Price_per_m2'].mean().idxmin() if not valid_area.empty else "N/A"
        else:
            cheapest_district = "N/A"

        c1.metric("Số nhà đang bán", f"{num_houses:,}")
        c2.metric("Giá trung bình", f"{avg_price:,.2f} Triệu")
        c3.metric("Khu vực rẻ nhất (m²)", f"{cheapest_district}")
        c4.metric("Căn đắt nhất", f"{max_price:,.2f} Triệu")

    st.divider()
    st.subheader(" Tableau Visualization")
    # (Giữ nguyên code nhúng Tableau của bạn)
    tableau_code = """
    <div class='tableauPlaceholder' id='viz1763483099173' style='position: relative'><noscript><a href='#'><img alt='tk ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Book7_17631271401140&#47;tk' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1763483099173');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                                            vizElement.parentNode.insertBefore(scriptElement, vizElement);                
                                            </script>
    """
    components.html(tableau_code, height=650, scrolling=True)

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
        price_range = st.slider("Khoảng giá (Trieu)", 100.0, 99999.0, (100.0, 99999.0))
    
    filtered_df = df.copy()
    if COL_PRICE in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df[COL_PRICE] >= price_range[0]) & (filtered_df[COL_PRICE] <= price_range[1])]
    
    if search_term and COL_DISTRICT in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df[COL_DISTRICT].str.contains(search_term, case=False, na=False) | 
            filtered_df[COL_TYPE].str.contains(search_term, case=False, na=False)
        ]

    st.info(f"Hiển thị {len(filtered_df)} bản ghi.")
    edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True)

    if st.button("💾 Lưu thay đổi"):
        st.session_state.df = edited_df
        st.success("Đã lưu dữ liệu!")

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
# MODULE 4: DỰ BÁO GIÁ (UPDATE CHO MODEL MỚI)
# =========================================================
# Giả định: Đoạn code này nằm trong khối `elif selected == "Dự báo Giá nhà":`

st.title(" 🏠 Dự báo Giá trị Bất động sản (chỉ mang tính chất tham khảo)")
st.markdown("---")

# 1. HÀM LOAD MODEL VÀ CỘT (CACHE ĐỂ TĂNG TỐC)
@st.cache_resource
def load_model_assets():
    try:
        # Load Model
        # Cần đảm bảo file này tồn tại
        model = joblib.load('house_price_model.pkl')
        
        # Load danh sách cột (Features)
        # Cần đảm bảo file này tồn tại
        cols = joblib.load('model_columns.pkl')
        
        return model, cols
    except Exception as e:
        st.error(f"Lỗi không tìm thấy file model: {e}")
        return None, None

model, model_columns = load_model_assets()

if model is None:
    st.warning("Vui lòng đảm bảo 2 file `house_price_model.pkl` và `model_columns.pkl` nằm cùng thư mục với `app.py`.")
    # Đặt st.stop() ở đây để dừng nếu model không load được
    # st.stop()

# --- Định nghĩa data mapping (Đảm bảo các biến này được định nghĩa) ---

# 2. TỰ ĐỘNG TRÍCH XUẤT DANH SÁCH LỰA CHỌN TỪ MODEL COLUMNS
# (Sử dụng các danh sách bạn đã cung cấp)

# Danh sách Quận/Huyện (Prefix: 'Quận_')
districts = [
    "Chương Mỹ", "Gia Lâm", "Hoài Đức", "Mê Linh", "Mỹ Đức", "Phú Xuyên", 
    "Phúc Thọ", "Quốc Oai", "Sóc Sơn", "Thanh Oai", "Thanh Trì", "Thường Tín", 
    "Thạch Thất", "Đan Phượng", "Đông Anh", 
    # Các Quận
    "Ba Đình", "Bắc Từ Liêm", "Cầu Giấy", "Hai Bà Trưng", "Hoàn Kiếm", "Hoàng Mai", 
    "Hà Đông", "Long Biên", "Nam Từ Liêm", "Thanh Xuân", "Tây Hồ", "Đống Đa", 
    "Thị xã Sơn Tây"
]

# Danh sách Phường/Xã (Đã rút gọn để code dễ đọc hơn, giữ nguyên nội dung bạn cung cấp)
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
    
    "Unknown": ["Unknown"] # Giữ lại Unknown nếu bạn muốn hiển thị
}

# Loại hình nhà (Prefix: 'Loại hình nhà ở_')
# Chỉ trích xuất nếu model_columns đã được load thành công
if model_columns is not None:
    house_types = sorted([c.replace('Loại hình nhà ở_', '') for c in model_columns if c.startswith('Loại hình nhà ở_')])
    
    # Pháp lý (Prefix: 'Giấy tờ pháp lý_')
    legal_types = sorted([c.replace('Giấy tờ pháp lý_', '') for c in model_columns if c.startswith('Giấy tờ pháp lý_')])
else:
    # Giá trị mặc định nếu load model thất bại
    house_types = ["Nhà mặt phố, mặt tiền", "Nhà ngõ, hẻm", "Nhà phố liền kề", "Unknown"]
    legal_types = ["Đã có sổ", "Đang chờ sổ", "Unknown"]

# 3. FORM NHẬP LIỆU
with st.form("prediction_form"):
    st.subheader("📋 Thông tin Bất động sản")
    
    # Hàng 1: Thông số kích thước
    c1, c2, c3 = st.columns(3)
    with c1:
        dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=10000.0, value=50.0, step=1.0)
        chieu_rong = st.number_input("Chiều Rộng / Mặt tiền (m)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)
    with c2:
        chieu_dai = st.number_input("Chiều Dài (m)", min_value=1.0, max_value=200.0, value=10.0, step=0.5)
        so_tang = st.number_input("Số tầng", min_value=1, max_value=100, value=3, step=1)
    with c3:
        so_phong = st.number_input("Số phòng ngủ", min_value=1, max_value=50, value=3, step=1)
        # Ngày tháng mặc định là hiện tại
        now = datetime.now()
        nam_gd = st.number_input("Năm giao dịch", value=now.year)
        thang_gd = st.number_input("Tháng giao dịch", min_value=1, max_value=12, value=now.month)

    st.markdown("---")
    st.subheader("📍 Vị trí & Phân loại")
    # Khởi tạo trạng thái ban đầu nếu chưa có
    if 'ward_enabled' not in st.session_state:
        st.session_state.ward_enabled = False

# Hàm callback để thay đổi trạng thái
def toggle_ward_state():
    # Gán giá trị của checkbox vào biến ward_enabled trong session state
    st.session_state.ward_enabled = st.session_state.ward_checkbox
    if 'ward_enabled' not in st.session_state:
        st.session_state.ward_enabled = False
    
    # Hàng 2: Vị trí và Loại hình (Đã di chuyển vào trong form)
    # Hàng 2: Vị trí và Loại hình (Nằm trong khối with st.form)
    c4, c5 = st.columns(2)
    with c4:
        # 1. Chọn Quận
        selected_district = st.selectbox("Quận / Huyện", districts)
        
        # 2. Lọc danh sách Phường/Xã
        filtered_wards = wards_map.get(selected_district, ["Không tìm thấy Phường/Xã"])
        
        # 3. Sử dụng KEY, ON_CHANGE để buộc cập nhật trạng thái (RẤT QUAN TRỌNG)
        st.checkbox(
            "Chọn Phường/Xã cụ thể?", 
            value=False, 
            key='ward_checkbox', 
            on_change=toggle_ward_state # GỌI HÀM KHI GIÁ TRỊ THAY ĐỔI
        )
        
        # Dùng biến mới đã được cập nhật bởi hàm callback để kiểm tra trạng thái
        st.write(f"Trạng thái ô kiểm: {st.session_state.ward_enabled}")
        
        # 4. Sử dụng trạng thái đã được buộc cập nhật để điều khiển disabled
        selected_ward = st.selectbox(
            "Phường / Xã", 
            filtered_wards, 
            # SỬ DỤNG session_state.ward_enabled
            disabled= not st.session_state.ward_enabled 
        )
    
    with c5:
        # Chọn Loại hình nhà ở
        selected_type = st.selectbox("Loại hình nhà ở", house_types)
        
        # Chọn Giấy tờ pháp lý
        selected_legal = st.selectbox("Giấy tờ pháp lý", legal_types)
        
    # --- KHẮC PHỤC LỖI NAMERROR: ĐỊNH NGHĨA NÚT SUBMIT TRONG FORM ---
    submit_btn = st.form_submit_button("💰 Dự đoán Giá Nhà", type="primary")

# 4. XỬ LÝ KHI ẤN NÚT DỰ BÁO (Đã sửa lỗi NameError)
if submit_btn:
    if model is None:
        st.error("Model không được load. Không thể thực hiện dự đoán.")
        st.stop()
        
    # A. Tạo DataFrame chứa đúng các cột mà Model yêu cầu, ban đầu gán bằng 0
    # Đảm bảo model_columns đã được load
    if model_columns is not None:
        input_data = pd.DataFrame(index=[0], columns=model_columns).fillna(0)
    else:
        st.error("Không tìm thấy danh sách cột của Model. Không thể dự đoán.")
        st.stop()

    # B. Gán giá trị số (Numeric)
    try:
        input_data['Diện tích'] = dien_tich
        input_data['Dài'] = chieu_dai
        input_data['Rộng'] = chieu_rong
        input_data['Số tầng'] = so_tang
        input_data['Số phòng ngủ'] = so_phong
        input_data['Năm'] = nam_gd
        input_data['Tháng'] = thang_gd
    except KeyError as e:
        st.error(f"Lỗi tên cột số liệu: {e}. Hãy kiểm tra lại tên cột trong dữ liệu train.")
        st.stop()

    # C. Gán giá trị One-Hot (Categorical)
    def set_one_hot(prefix, value):
        col_name = f"{prefix}{value}"
        if col_name in input_data.columns:
            input_data[col_name] = 1
    
    # Kích hoạt các cột tương ứng
    # Lưu ý: Các biến selected_type, selected_legal đã được định nghĩa
    set_one_hot('Quận_', selected_district)
    set_one_hot('Loại hình nhà ở_', selected_type)
    set_one_hot('Giấy tờ pháp lý_', selected_legal)
    
    if st.session_state.ward_checkbox: # Sử dụng trạng thái đã được đồng bộ
        set_one_hot('Huyện_', selected_ward)

    # D. Thực hiện dự đoán
    with st.spinner("Đang tính toán..."):
        try:
            predicted_price = model.predict(input_data)[0]
            
            # Hiển thị kết quả đẹp mắt
            st.success("✅ Dự báo thành công!")
            
            # Cần xử lý giá trị dự đoán nếu nó quá nhỏ (do lỗi logarit hoặc model chưa tốt)
            if predicted_price < 0:
                predicted_price = 0.1 # Giả sử mức giá tối thiểu
                st.warning("Giá trị dự đoán âm. Đã điều chỉnh về 0.1 Tỷ.")
            
            metric_col1, metric_col2 = st.columns([2, 1])
            with metric_col1:
                st.markdown(f"""
                <div style="background-color: #e6fffa; padding: 20px; border-radius: 10px; border: 2px solid #38b2ac; text-align: center;">
                    <h3 style="color: #2c7a7b; margin:0;">GIÁ TRỊ ƯỚC TÍNH</h3>
                    <h1 style="color: #285e61; font-size: 48px; margin: 10px 0;">{predicted_price:,.2f} Tỷ</h1>
                    <p style="color: #4a5568;">~ {(predicted_price * 1_000_000_000 / dien_tich):,.0f} VNĐ / m²</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.info("Thông tin đầu vào:")
                st.write(f"- **Diện tích:** {dien_tich} m²")
                st.write(f"- **Vị trí:** {selected_district} {'/ ' + selected_ward if st.session_state.ward_checkbox else ''}")
                st.write(f"- **Loại:** {selected_type}")
                st.write(f"- **Pháp lý:** {selected_legal}")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình tính toán: {str(e)}")
            with st.expander("Chi tiết lỗi (Dành cho Dev)"):
                st.code(e)
                # Hiển thị DataFrame đầu vào để dễ debug
                st.write("DataFrame đầu vào (Input Data):")
                st.dataframe(input_data)

