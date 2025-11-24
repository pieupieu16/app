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

## =========================================================
# MODULE 4: DỰ BÁO GIÁ (UPDATE CHO MODEL MỚI)
# =========================================================
elif selected == "Dự báo Giá nhà":
    st.title(" Dự báo Giá trị Bất động sản( chỉ mang tính chất tham khảo )")
    st.markdown("---")

    # 1. HÀM LOAD MODEL VÀ CỘT (CACHE ĐỂ TĂNG TỐC)
    @st.cache_resource
    def load_model_assets():
        try:
            # Load Model
            model = joblib.load('house_price_model.pkl')
            
            # Load danh sách cột (Features)
            cols = joblib.load('model_columns.pkl')
            
            return model, cols
        except Exception as e:
            st.error(f"Lỗi không tìm thấy file model: {e}")
            return None, None

    model, model_columns = load_model_assets()

    if model is None:
        st.warning("Vui lòng đảm bảo 2 file `house_price_model.pkl` và `model_columns.pkl` nằm cùng thư mục với `app.py`.")
        st.stop()

    # 2. TỰ ĐỘNG TRÍCH XUẤT DANH SÁCH LỰA CHỌN TỪ MODEL COLUMNS
    # Logic: Lọc các cột One-Hot (bắt đầu bằng prefix) để đưa vào Selectbox
    
    # Danh sách Quận/Huyện (Prefix: 'Quận_')
    districts = sorted([c.replace('Quận_', '') for c in model_columns if c.startswith('Quận_')])
    
    # Danh sách Phường/Xã (Prefix: 'Huyện_') - Lưu ý: Trong dữ liệu của bạn 'Huyện_' thực chất là tên Phường
    wards = sorted([c.replace('Huyện_', '') for c in model_columns if c.startswith('Huyện_')])
    
    # Loại hình nhà (Prefix: 'Loại hình nhà ở_')
    house_types = sorted([c.replace('Loại hình nhà ở_', '') for c in model_columns if c.startswith('Loại hình nhà ở_')])
    
    # Pháp lý (Prefix: 'Giấy tờ pháp lý_')
    legal_types = sorted([c.replace('Giấy tờ pháp lý_', '') for c in model_columns if c.startswith('Giấy tờ pháp lý_')])

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
        
        # Hàng 2: Vị trí và Loại hình
        c4, c5 = st.columns(2)
        with c4:
            # Chọn Quận
            selected_district = st.selectbox("Quận / Huyện", districts)
            
            # Chọn Phường (Optional: Có thể lọc phường theo quận nếu có data mapping, ở đây show all)
            use_ward = st.checkbox("Chọn Phường/Xã cụ thể?", value=False)
            selected_ward = st.selectbox("Phường / Xã", wards, disabled= use_ward)
            
        with c5:
            selected_type = st.selectbox("Loại hình nhà ở", house_types)
            selected_legal = st.selectbox("Giấy tờ pháp lý", legal_types)

        # Nút Submit
        submit_btn = st.form_submit_button("🚀 DỰ BÁO GIÁ NGAY", use_container_width=True)

    # 4. XỬ LÝ KHI ẤN NÚT DỰ BÁO
    if submit_btn:
        # A. Tạo DataFrame chứa đúng các cột mà Model yêu cầu, ban đầu gán bằng 0
        input_data = pd.DataFrame(index=[0], columns=model_columns).fillna(0)

        # B. Gán giá trị số (Numeric)
        # Lưu ý: Tên cột phải khớp CHÍNH XÁC với file model_columns.pkl (Dựa trên log bạn cung cấp)
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
        # Hàm helper để set giá trị 1 cho cột One-hot
        def set_one_hot(prefix, value):
            col_name = f"{prefix}{value}"
            if col_name in input_data.columns:
                input_data[col_name] = 1
        
        # Kích hoạt các cột tương ứng
        set_one_hot('Quận_', selected_district)
        set_one_hot('Loại hình nhà ở_', selected_type)
        set_one_hot('Giấy tờ pháp lý_', selected_legal)
        
        if use_ward:
            set_one_hot('Huyện_', selected_ward)

        # D. Thực hiện dự đoán
        with st.spinner("Đang tính toán..."):
            try:
                predicted_price = model.predict(input_data)[0]
                
                # Hiển thị kết quả đẹp mắt
                st.success("✅ Dự báo thành công!")
                
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
                    st.write(f"- **Vị trí:** {selected_district}")
                    st.write(f"- **Loại:** {selected_type}")

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình tính toán: {str(e)}")
                # Mở rộng để debug nếu cần
                with st.expander("Chi tiết lỗi (Dành cho Dev)"):
                    st.write(e)
                    st.write("Danh sách cột đầu vào:", input_data.columns.tolist())