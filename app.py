import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
import io
# 1. THÊM THƯ VIỆN MENU
from streamlit_option_menu import option_menu 

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hanoi Real Estate Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed" # Ẩn sidebar đi
)

# --- CSS TÙY CHỈNH (Giữ nguyên) ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {
        background-color: #ffffff !important; 
        border: 1px solid #e6e6e6; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stMetricLabel {color: #6c757d !important;} 
    .stMetricValue {color: #000000 !important;} 
    .stMetric div, .stMetric p {color: #000000 !important;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. XỬ LÝ DỮ LIỆU (Giữ nguyên hàm của bạn) ---
@st.cache_data
def load_data_v2():
    file_path = 'dự tính giá nhà - Trang tính1 (2).csv'
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        rename_mapping = {
            'Giá(ty)': 'Giá (Tỷ)', 'Diện Tích(m2)': 'Diện tích (m2)',
            'numberbedroom': 'Phòng ngủ', 'numberbathroom': 'Phòng tắm',
            'Loại Hình(căn hộ ,nhà,villa)': 'Loại nhà',
            'KHoảng cách đến trung tâm (Km)': 'Khoảng cách trung tâm (Km)',
            'sổ đỏ': 'Sổ đỏ', 'Hướng Nhà': 'Hướng nhà'
        }
        df.rename(columns=rename_mapping, inplace=True)
        
        cols_to_numeric = ['Giá (Tỷ)', 'Diện tích (m2)', 'Phòng ngủ', 'Khoảng cách trung tâm (Km)']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['Giá (Tỷ)', 'Diện tích (m2)'], inplace=True)
        
        # Làm sạch giá (Giữ nguyên logic của bạn)
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].astype(str).str.strip()
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].str.replace('tỷ', '', regex=False).str.replace('ty', '', regex=False).str.replace(' ', '', regex=False)
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].str.replace(r'[^\d.]', '', regex=True) 
        df['Giá (Tỷ)'] = pd.to_numeric(df['Giá (Tỷ)'], errors='coerce')
        df['Diện tích (m2)'] = pd.to_numeric(df['Diện tích (m2)'], errors='coerce')

        # Gộp cột quận (Giữ nguyên logic của bạn)
        quan_columns = ['Ba Đình', 'Cầu Giấy', 'Đống Đa', 'Hai Bà Trưng', 'Thanh Xuân', 
                        'Hoàng Mai', 'Long Biên', 'Hà Đông', 'Tây Hồ', 'Nam Từ Liêm', 
                        'Bắc Từ Liêm', 'Thanh Trì']
        valid_quan_cols = [q for q in quan_columns if q in df.columns]

        if not valid_quan_cols:
            df['Quận'] = "Chưa xác định"
        else:
            def get_quan(row):
                for q in valid_quan_cols:
                    if row.get(q) == 1.0:
                        return q
                return "Khác"
            df['Quận'] = df.apply(get_quan, axis=1)

        # Tổng tiện ích (Giữ nguyên logic của bạn)
        tien_ich = ['sercurity(1 or 0)', 'Giải trí(1 or 0)', 'Giao thông(1 or 0)', 
                    'Bệnh viện(1 or 0)', 'Market(1 or 0)', 'Giáo dục(1 or 0)']
        valid_tien_ich = [t for t in tien_ich if t in df.columns]
        if valid_tien_ich:
            df['Tổng tiện ích'] = df[valid_tien_ich].sum(axis=1)
        else:
            df['Tổng tiện ích'] = 0

        return df
    
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return pd.DataFrame()

# --- KHỞI TẠO DỮ LIỆU ---
if 'data' not in st.session_state:
    st.session_state['data'] = load_data_v2()

df = st.session_state['data']

# KIỂM TRA AN TOÀN
if df.empty:
    st.warning("Chưa có dữ liệu. Vui lòng kiểm tra file CSV.")
    st.stop()

# --- 2. THAY THẾ SIDEBAR BẰNG MENU NGANG ---
# Bỏ hoàn toàn 'with st.sidebar:'
menu = option_menu(
    menu_title=None, # Bắt buộc
    options=["Trang chủ & Định giá", "Phân tích Dữ liệu", "Quản lý Dữ liệu", "Tableau"], # Đổi tên
    icons=["house-door", "graph-up", "database-gear", "bar-chart-line"], # Icon
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal", # ĐÂY LÀ CHÌA KHÓA
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
)

# --- MODULE 1: TRANG CHỦ & ĐỊNH GIÁ (Thiết kế lại) ---
if menu == "Trang chủ & Định giá":
    
    # A. PHẦN TIÊU ĐỀ
    st.title("Xác định giá trị bất động sản nhanh và chính xác nhất")
    st.markdown("Sử dụng dữ liệu lớn để phân tích và dự đoán giá nhà tại Hà Nội.")

    # B. PHẦN CÔNG CỤ ĐỊNH GIÁ (Mô phỏng ảnh b62a62)
    # Dùng st.tabs để tạo các tab "Căn hộ chung cư", "Officetel"...
    tab_chungcu, tab_officetel, tab_bietthu = st.tabs(["Căn hộ chung cư", "Officetel", "Biệt thự/Shophouse"])

    with tab_chungcu:
        st.subheader("Định giá Căn hộ chung cư")
        
        # Dùng st.columns để tạo layout lưới cho bộ lọc
        col1, col2, col3 = st.columns(3)
        with col1:
            tinh_thanh = st.selectbox("Tỉnh/Thành phố", ["Hà Nội", "TP. Hồ Chí Minh"], key="t1")
            quan_huyen = st.selectbox("Quận/Huyện", df['Quận'].unique(), key="q1")
        with col2:
            du_an = st.selectbox("Dự án", ["Vinhomes Smart City", "Vinhomes Ocean Park", "Khác"], key="d1")
            toa_nha = st.selectbox("Tòa nhà", ["S1.01", "S1.02", "G1", "G2"], key="tna1")
        with col3:
            tang = st.number_input("Tầng", min_value=1, max_value=50, value=10, key="ta1")
            ma_can = st.text_input("Mã căn (Nếu có)", key="mc1")
        
        if st.button("Định giá ngay", type="primary", key="b1"):
            # (Thêm logic định giá của bạn ở đây)
            st.success("Đang xử lý định giá...")

    with tab_officetel:
        st.subheader("Định giá Officetel")
        # (Thêm các bộ lọc tương tự cho Officetel...)
        st.write("Các bộ lọc cho Officetel...")

    with tab_bietthu:
        st.subheader("Định giá Biệt thự / Shophouse")
        # (Thêm các bộ lọc tương tự...)
        st.write("Các bộ lọc cho Biệt thự...")

    st.divider() # Ngăn cách

    # C. PHẦN CHỈ SỐ (Metrics) - (Giống ảnh b62a62)
    st.subheader("Thống kê thị trường")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số tin đăng", f"{len(df):,}")
    col2.metric("Giá trung bình (Toàn thị trường)", f"{df['Giá (Tỷ)'].mean():.2f} Tỷ")
    col3.metric("Diện tích trung bình", f"{df['Diện tích (m2)'].mean():.1f} m²")

    # D. CÁC BIỂU ĐỒ (Lấy từ Dashboard cũ của bạn)
    st.subheader("Tổng quan thị trường")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Phân bổ Giá theo Diện tích")
        fig_map = px.scatter(df, x="Diện tích (m2)", y="Giá (Tỷ)", color="Quận", size="Giá (Tỷ)")
        st.plotly_chart(fig_map, use_container_width=True)
    with c2:
        st.markdown("##### Tỷ lệ Loại hình nhà")
        df['Loại nhà'] = df['Loại nhà'].astype(str).str.strip()
        fig_pie = px.pie(df, names='Loại nhà', title='Cơ cấu nguồn cung', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)


# --- MODULE 2: PHÂN TÍCH DỮ LIỆU (Ghép 2 module cũ) ---
elif menu == "Phân tích Dữ liệu":
    st.title("📈 Phân tích & Trực quan hóa Chuyên sâu")

    # Mô phỏng "Mega-Menu" (ảnh b62ac2) bằng st.expander
    with st.expander("Bộ lọc Phân tích (Phân tích khu vực & dự án)"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("#### Phân tích khu vực")
            # Cho phép chọn nhiều quận để phân tích
            quan_filter_list = st.multiselect("Chọn Quận/Huyện:", options=df['Quận'].unique(), default=df['Quận'].unique()[:3])
        with col_f2:
            st.markdown("#### Phân tích Loại nhà")
            loai_nha_list = st.multiselect("Chọn Loại nhà:", options=df['Loại nhà'].unique(), default=df['Loại nhà'].unique())
    
    # Lọc df dựa trên lựa chọn
    df_filtered = df[df['Quận'].isin(quan_filter_list) & df['Loại nhà'].isin(loai_nha_list)]

    if df_filtered.empty:
        st.warning("Không có dữ liệu với bộ lọc hiện tại.")
    else:
        st.subheader(f"Kết quả phân tích cho {len(df_filtered)} BĐS")
        st.divider()

        st.subheader("1. Tương quan: Giá & Diện tích")
        fig1 = px.scatter(df_filtered, x="Diện tích (m2)", y="Giá (Tỷ)", color="Quận", 
                            size="Tổng tiện ích", trendline="ols")
        st.plotly_chart(fig1, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("2. Top Quận đắt đỏ nhất (Đã lọc)")
            avg_price = df_filtered.groupby('Quận')['Giá (Tỷ)'].mean().sort_values(ascending=False).reset_index()
            fig2 = px.bar(avg_price, x='Quận', y='Giá (Tỷ)', color='Giá (Tỷ)')
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.subheader("3. Phân phối giá theo Loại nhà (Đã lọc)")
            fig3 = px.box(df_filtered, x="Loại nhà", y="Giá (Tỷ)", color="Loại nhà") 
            st.plotly_chart(fig3, use_container_width=True)

        # (Giữ nguyên các biểu đồ khác của bạn...)


# --- MODULE 3: QUẢN LÝ DỮ LIỆU (Ghép 2 module cũ) ---
elif menu == "Quản lý Dữ liệu":
    st.title("🗃️ Trung tâm Quản lý & Làm sạch Dữ liệu")

    # 1. Phần Làm sạch
    st.subheader("✨ Data Refinery (Làm sạch)")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Thống kê dữ liệu thiếu (Null)")
        cols_exist = [c for c in ['Giá (Tỷ)', 'Diện tích (m2)', 'Phòng ngủ', 'Sổ đỏ'] if c in df.columns]
        null_counts = df[cols_exist].isnull().sum()
        st.dataframe(null_counts)
    with col2:
        st.info("Công cụ xử lý")
        if 'Phòng ngủ' in df.columns:
            if st.button("Điền số 'Phòng ngủ' bị thiếu bằng Median"):
                df['Phòng ngủ'] = df['Phòng ngủ'].fillna(df['Phòng ngủ'].median())
                st.session_state['data'] = df
                st.success("Đã xử lý xong!")
                st.rerun()
        
        threshold = st.number_input("Giá trần lọc ngoại lai (Tỷ):", value=500.0, step=10.0)
        if st.button("Loại bỏ ngoại lai"):
            df = df[df['Giá (Tỷ)'] <= threshold]
            st.session_state['data'] = df
            st.warning("Đã loại bỏ ngoại lai!")
            st.rerun()

    st.divider()

    # 2. Phần CRUD
    st.subheader("✏️ Xem & Chỉnh sửa Dữ liệu")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if st.button("Lưu thay đổi tạm thời"):
        st.session_state['data'] = edited_df
        st.success("Đã cập nhật!")

    st.divider()

    # 3. Phần Tải lên / Tải xuống
    st.subheader("📥 Tải lên / Tải xuống")
    c_up, c_down = st.columns(2)
    with c_up:
        uploaded_file = st.file_uploader("Tải lên file CSV/Excel khác", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            # (Logic tải lên của bạn...)
            st.success("Tải dữ liệu mới thành công!")
            st.rerun()
    with c_down:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Tải xuống CSV", csv, "data_final.csv", "text/csv")
        # (Logic xuất Excel của bạn...)


# --- MODULE 4: TABLEAU INTEGRATION ---
elif menu == "Tableau":
    st.title("🌐 Kết nối Tableau")
    st.markdown("""
    Đây là khu vực tích hợp Dashboard từ Tableau Public. 
    Bạn có thể tương tác (Lọc, Zoom, Click) trực tiếp ngay tại đây.
    """)
    
    # (Giữ nguyên code nhúng Tableau của bạn)
    tableau_html_code = """
    <div class='tableauPlaceholder' id='viz1763127239393' style='position: relative'><noscript><a href='#'><img alt='tk ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='httpsD%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Book7_17631271401140&#47;tk' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1763127239393');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
    components.html(tableau_html_code, height=850, scrolling=True)