import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
import io
import joblib
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
        
        # 1. Làm sạch tên cột
        df.columns = df.columns.str.strip()
        
        # 2. Đổi tên cột (Mapping)
        rename_mapping = {
            'Giá(ty)': 'Giá (Tỷ)',
            'Diện Tích(m2)': 'Diện tích (m2)',
            'numberbedroom': 'Phòng ngủ',
            'numberbathroom': 'Phòng tắm',
            'Loại Hình(căn hộ ,nhà,villa)': 'Loại nhà', # <-- Cột này sẽ được lọc
            'KHoảng cách đến trung tâm (Km)': 'Khoảng cách trung tâm (Km)',
            'sổ đỏ': 'Sổ đỏ',
            'Hướng Nhà': 'Hướng nhà'
        }
        df.rename(columns=rename_mapping, inplace=True)

        # 3. Ép kiểu dữ liệu số
        cols_to_numeric = ['Giá (Tỷ)', 'Diện tích (m2)', 'Phòng ngủ', 'Khoảng cách trung tâm (Km)']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Xóa dòng không có giá hoặc diện tích
        df.dropna(subset=['Giá (Tỷ)', 'Diện tích (m2)'], inplace=True)

        # ... (Phần code làm sạch cột 'Giá (Tỷ)' và 'Diện tích (m2)' của bạn)
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].astype(str).str.strip()
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].str.replace('tỷ', '', regex=False).str.replace('ty', '', regex=False).str.replace(' ', '', regex=False)
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].str.replace(r'[^\d.]', '', regex=True) 
        df['Giá (Tỷ)'] = pd.to_numeric(df['Giá (Tỷ)'], errors='coerce')
        df['Diện tích (m2)'] = pd.to_numeric(df['Diện tích (m2)'], errors='coerce')
        
        # --- 🟢 THÊM CHỨC NĂNG LỌC 'LOẠI NHÀ' (MỚI) ---
        if 'Loại nhà' in df.columns:
            # 1. Chuẩn hóa (xóa khoảng trắng thừa và chuyển thành chữ thường cho chắc)
            df['Loại nhà'] = df['Loại nhà'].astype(str).str.strip().str.lower()
            
            # 2. Danh sách các giá trị được phép
            allowed_loai_nha = ['căn hộ', 'nhà', 'villa']
            
            # 3. Lọc DataFrame (chỉ giữ lại các hàng có giá trị trong danh sách)
            df = df[df['Loại nhà'].isin(allowed_loai_nha)].copy()
        # --- KẾT THÚC PHẦN MỚI ---

        # 4. GỘP CỘT QUẬN (Giữ nguyên logic của bạn)
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

        # 5. Tổng tiện ích (Giữ nguyên logic của bạn)
        tien_ich = ['sercurity(1 or 0)', 'Giải trí(1 or 0)', 'Giao thông(1 or 0)', 
                    'Bệnh viện(1 or 0)', 'Market(1 or 0)', 'Giáo dục(1 or 0)']
        valid_tien_ich = [t for t in tien_ich if t in df.columns]
        if valid_tien_ich:
            df['Tổng tiện ích'] = df[valid_tien_ich].sum(axis=1)
        else:
            df['Tổng tiện ích'] = 0

        return df
    
    except Exception as e:
        # Sửa lỗi này để hiển thị rõ hơn trên Streamlit
        st.error(f"Lỗi khi đọc file CSV: {e}")
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

# ===================================================================
# --- MODULE 1: TRANG CHỦ & ĐỊNH GIÁ (ĐÃ THIẾT KẾ LẠI HOÀN TOÀN) ---
# ===================================================================
if menu == "Trang chủ & Định giá":
    
    st.title("🤖 Công cụ Định giá Bất động sản Hà Nội")
    st.markdown("Nhập các thông số của bất động sản để dự đoán giá trị (Tỷ VNĐ).")
    @st.cache_resource # Dùng cache_resource cho model
    def load_model(model_path="model.pkl"):
        try:
            model = joblib.load(model_path)
            return model
        except FileNotFoundError:
            st.error(f"Lỗi: Không tìm thấy file model '{model_path}'.")
            st.error("Vui lòng đảm bảo file model (ví dụ: model.pkl) nằm cùng thư mục với app.py")
            return None
        except Exception as e:
            st.error(f"Lỗi khi tải model: {e}")
            return None

    # Tải model khi khởi động
    model = load_model()

    # Kiểm tra xem model đã được tải chưa
    if model is None:
        st.warning("Mô hình dự đoán hiện chưa sẵn sàng. Vui lòng kiểm tra file model.")
        st.stop() # Dừng chạy Module này nếu không có model

    # --- DANH SÁCH CÁC INPUT (Từ yêu cầu của bạn) ---
    # Đây là các danh sách để tạo input (giao diện)
    # Rất quan trọng: Tên cột one-hot (quan_list) phải khớp 100% với tên feature trong model
    quan_list = ['Ba Đình', 'Cầu Giấy', 'Đống Đa', 'Hai Bà Trưng', 'Thanh Xuân', 
                 'Hoàng Mai', 'Long Biên', 'Hà Đông', 'Tây Hồ', 'Nam Từ Liêm', 
                 'Bắc Từ Liêm', 'Thanh Trì']
    
    loai_hinh_list = ['căn hộ', 'nhà', 'villa'] # (Từ input của bạn)
    
    # Giả định các hướng nhà (Bạn có thể cần sửa lại)
    huong_nha_list = ['KXĐ', 'Đông', 'Tây', 'Nam', 'Bắc', 'Đông Nam', 'Tây Nam', 'Đông Bắc', 'Tây Bắc'] 

    # --- B. FORM NHẬP LIỆU ---
    # st.form giúp nhóm tất cả input và chỉ gửi khi bấm nút
    with st.form(key="prediction_form"):
        
        st.subheader("Thông tin cơ bản")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dien_tich = st.number_input("Diện Tích (m2)", min_value=10.0, value=50.0, step=1.0)
            phong_ngu = st.number_input("Số phòng ngủ (numberbedroom)", min_value=0, value=2, step=1)
            phong_tam = st.number_input("Số phòng tắm (numberbathroom)", min_value=0, value=2, step=1)
        
        with col2:
            so_tang = st.number_input("Số tầng", min_value=1, value=1, step=1)
            mat_tien = st.number_input("Mặt tiền (m)", min_value=0.0, value=5.0, step=0.1)
            khoang_cach_tt = st.number_input("Khoảng cách đến trung tâm (Km)", min_value=0.0, value=5.0, step=0.1)

        with col3:
            # Giao diện nhập Quận (UI)
            # Chúng ta dùng 1 selectbox cho dễ dùng, sau đó sẽ tự one-hot
            quan_input = st.selectbox("Chọn Quận", quan_list)
            loai_hinh_input = st.selectbox("Loại Hình", loai_hinh_list)
            huong_nha_input = st.selectbox("Hướng Nhà", huong_nha_list)

        st.subheader("Thông tin pháp lý & tiện ích (1=Có, 0=Không)")
        col4, col5, col6 = st.columns(3)

        # Dùng st.radio cho các biến nhị phân (1/0)
        with col4:
            noi_that = st.radio("Nội thất", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
            so_do = st.radio("Sổ đỏ", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
            security = st.radio("An ninh (sercurity)", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
        
        with col5:
            giai_tri = st.radio("Giải trí", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
            giao_thong = st.radio("Giao thông", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
            benh_vien = st.radio("Bệnh viện", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
        
        with col6:
            market = st.radio("Chợ/Siêu thị (Market)", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)
            giao_duc = st.radio("Giáo dục", [1, 0], format_func=lambda x: "Có" if x == 1 else "Không", horizontal=True)

        # Nút dự đoán
        submit_button = st.form_submit_button(label="DỰ ĐOÁN GIÁ", use_container_width=True)

    # --- C. XỬ LÝ VÀ DỰ ĐOÁN (Sau khi bấm nút) ---
    if submit_button:
        try:
            # 1. Tạo một dictionary để chứa tất cả dữ liệu
            input_data = {}

            # 2. Thêm các feature số và nhị phân (đã nhập)
            input_data['Diện Tích(m2)'] = dien_tich
            input_data['numberbedroom'] = phong_ngu
            input_data['numberbathroom'] = phong_tam
            input_data['Số tầng'] = so_tang
            input_data['Nội thất (1/0)'] = noi_that
            input_data['Mặt tiền'] = mat_tien
            input_data['sổ đỏ'] = so_do
            input_data['KHoảng cách đến trung tâm (Km)'] = khoang_cach_tt
            input_data['sercurity(1 or 0)'] = security
            input_data['Giải trí(1 or 0)'] = giai_tri
            input_data['Giao thông(1 or 0)'] = giao_thong
            input_data['Bệnh viện(1 or 0)'] = benh_vien
            input_data['Market(1 or 0)'] = market
            input_data['Giáo dục(1 or 0)'] = giao_duc
            
            # 3. Thêm các feature categorical (Giả định model của bạn chấp nhận string)
            # QUAN TRỌNG: Nếu model của bạn cần one-hot cho 'Loại Hình' và 'Hướng Nhà', 
            # bạn cần xử lý tương tự như 'Quận' bên dưới.
            input_data['Loại Hình(căn hộ ,nhà,villa)'] = loai_hinh_input
            input_data['Hướng Nhà'] = huong_nha_input

            # 4. Xử lý One-Hot Encoding cho Quận
            # Tạo 12 cột (Ba Đình, Cầu Giấy,...)
            for q in quan_list:
                input_data[q] = 1 if q == quan_input else 0

            # 5. Xác định thứ tự cột (CỰC KỲ QUAN TRỌNG)
            # Thứ tự này phải khớp 100% với thứ tự cột khi bạn huấn luyện model.
            # Hãy kiểm tra lại file notebook training của bạn để lấy thứ tự chính xác.
            
            # Dưới đây là thứ tự dựa trên danh sách bạn cung cấp:
            final_feature_columns = [
                'Diện Tích(m2)', 'numberbedroom', 'numberbathroom', 'Số tầng', 
                'Nội thất (1/0)', 'Mặt tiền', 'Loại Hình(căn hộ ,nhà,villa)', 'sổ đỏ', 
                'KHoảng cách đến trung tâm (Km)', 'sercurity(1 or 0)', 'Hướng Nhà',
                'Ba Đình', 'Cầu Giấy', 'Đống Đa', 'Hai Bà Trưng', 'Thanh Xuân', 
                'Hoàng Mai', 'Long Biên', 'Hà Đông', 'Tây Hồ', 'Nam Từ Liêm', 
                'Bắc Từ Liêm', 'Thanh Trì', 
                'Giải trí(1 or 0)', 'Giao thông(1 or 0)', 
                'Bệnh viện(1 or 0)', 'Market(1 or 0)', 'Giáo dục(1 or 0)'
            ]

            # 6. Tạo DataFrame 1 dòng
            # Đảm bảo dữ liệu được sắp xếp đúng thứ tự cột
            input_df = pd.DataFrame([input_data], columns=final_feature_columns)

            # 7. Dự đoán
            with st.spinner("Đang tính toán..."):
                prediction = model.predict(input_df)
                predicted_price = prediction[0] # Lấy kết quả dự đoán

            # 8. Hiển thị kết quả
            st.success(f"Dự đoán thành công!")
            st.metric(label="Giá trị Bất động sản (Ước tính)", 
                      value=f"{predicted_price:,.2f} Tỷ VNĐ")
            
            # (Tùy chọn) Hiển thị dữ liệu đã gửi cho model để debug
            with st.expander("Xem dữ liệu đầu vào đã xử lý"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình dự đoán:")
            st.error(e)
            st.error("Gợi ý: Hãy kiểm tra lại danh sách 'final_feature_columns' trong code xem đã khớp 100% với model chưa.")


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