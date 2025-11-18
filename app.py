import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hệ thống Quản lý & Định giá BĐS Hà Nội",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh giao diện cho đẹp hơn
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .stMetric {background-color: white; padding: 10px; border-radius: 8px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM XỬ LÝ DỮ LIỆU (CORE) ---
@st.cache_data
def load_data(file_path='augmented_housing_data.csv'):
    try:
        df = pd.read_csv(file_path)
        
        # 1. Xử lý tên cột (xóa khoảng trắng thừa)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # 2. XÓA CỘT TRÙNG (Quan trọng để sửa lỗi "Ambiguous")
        df = df.loc[:, ~df.columns.duplicated()]

        # 3. Tái tạo cột phân loại từ One-Hot Encoding (để hiển thị và vẽ biểu đồ)
        # Hàm nội bộ để gom nhóm One-Hot
        def reverse_ohe(row, prefix):
            cols = [c for c in df.columns if c.startswith(prefix)]
            for c in cols:
                if row[c] == 1:
                    return c.replace(prefix, '')
            return 'Khác'

        # Tạo cột 'Quận' nếu chưa có
        if 'Quận' not in df.columns:
            df['Quận'] = df.apply(lambda x: reverse_ohe(x, 'Dist_'), axis=1)
        
        # Tạo cột 'Loại nhà' nếu chưa có
        if 'Loại nhà' not in df.columns:
            df['Loại nhà'] = df.apply(lambda x: reverse_ohe(x, 'Type_'), axis=1)
            df['Loại nhà'] = df['Loại nhà'].str.lower()

        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

# --- 3. QUẢN LÝ STATE (Lưu trạng thái phiên làm việc) ---
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Biến tắt dùng chung
df = st.session_state.df

# --- 4. MENU ĐIỀU HƯỚNG (DẠNG NGANG) ---
# Lưu ý: Không đặt trong 'with st.sidebar:' nữa

# (Tùy chọn) Hiển thị Logo/Tiêu đề phía trên Menu
col_logo, col_text = st.columns([1, 5])
with col_logo:
    st.image("Gemini_Generated_Image_zgk17rzgk17rzgk1.png", width=60)
with col_text:
    st.title("Hệ thống Định giá BĐS Hà Nội")

# Tạo Menu ngang
selected = option_menu(
    menu_title=None,  # Để None cho menu ngang gọn hơn
    options=["Trang chủ & Tableau", "Quản lý Dữ liệu (CRUD)", "Phân tích Trực quan", "Dự báo Giá nhà"],
    icons=["house", "table", "bar-chart-line", "magic"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",  # <--- QUAN TRỌNG: Chuyển thành hàng ngang
    styles={
        "container": {"padding": "0!important", "background-color": "#a13d3d"},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"}, # Màu xanh trùng với nút dự báo
    }
    
)
# =========================================================
# MODULE 1: TRANG CHỦ & TABLEAU
# =========================================================
if selected == "Trang chủ & Tableau":
    st.title(" Dashboard Tổng quan & Tableau")
    st.markdown("Kết nối dữ liệu trực quan từ công cụ Tableau Public.")
    # CSS tùy chỉnh: Nền đen, Chữ xanh lá (Green Matrix Style)
    st.markdown("""
        <style>
        /* Áp dụng cho toàn bộ hộp Metric */
        [data-testid="stMetric"] {
            background-color: #000000 !important; /* Nền đen */
            border: 1px solid #00ff00; /* Viền xanh lá neon */
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.2); /* Phát sáng nhẹ */
        }
        
        /* Màu chữ cho Label (Tiêu đề nhỏ phía trên) */
        [data-testid="stMetricLabel"] p {
            color: #00ff00 !important; /* Xanh lá */
            font-weight: bold;
        }
        
        /* Màu chữ cho Value (Giá trị số to) */
        [data-testid="stMetricValue"] div {
            color: #00ff00 !important; /* Xanh lá */
            text-shadow: 0 0 5px #00ff00; /* Hiệu ứng phát sáng chữ */
        }
        </style>
        """, unsafe_allow_html=True)
    # Dashboard số liệu nhanh (Metric)
    # Dashboard số liệu nhanh (Metric)
    c1, c2, c3, c4 = st.columns(4)
    
    if not df.empty:
        # 1. Số nhà đang bán
        num_houses = len(df)
        
        # 2. Giá trung bình
        avg_price = df['Price_Billion'].mean()
        
        # 3. Khu vực có giá/m2 rẻ nhất (Logic phức tạp hơn xíu)
        # Tạo cột đơn giá tạm thời: Giá / Diện tích
        # Lưu ý: Tránh chia cho 0 bằng cách lọc area > 0
        valid_area = df[df['Area_m2'] > 0].copy()
        valid_area['Price_per_m2'] = valid_area['Price_Billion'] / valid_area['Area_m2']
        
        # Nhóm theo Quận và tính trung bình đơn giá, sau đó lấy Quận có giá thấp nhất
        if not valid_area.empty:
            cheapest_district = valid_area.groupby('Quận')['Price_per_m2'].mean().idxmin()
        else:
            cheapest_district = "N/A"
            
        # 4. Căn đắt nhất
        max_price = df['Price_Billion'].max()

        # --- HIỂN THỊ ---
        c1.metric("Số nhà đang bán", f"{num_houses:,}")
        c2.metric("Giá trung bình", f"{avg_price:,.2f} Tỷ")
        c3.metric("Khu vực rẻ nhất (theo m²)", f"{cheapest_district}")
        c4.metric("Căn đắt nhất", f"{max_price:,.2f} Tỷ")

    st.divider()
    
    # --- NHÚNG TABLEAU ---
    st.subheader(" Tableau Visualization")
    # Đây là mã nhúng mẫu (Bạn có thể thay bằng link Tableau của chính bạn)
    tableau_code = """
    <div class='tableauPlaceholder' id='viz1763483099173' style='position: relative'><noscript><a href='#'><img alt='tk ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Book7_17631271401140&#47;tk' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1763483099173');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                                            vizElement.parentNode.insertBefore(scriptElement, vizElement);                
                                            </script>

    """
    components.html(tableau_code, height=650, scrolling=True)

# =========================================================
# MODULE 2: QUẢN LÝ DỮ LIỆU (CRUD + TÌM KIẾM)
# =========================================================
elif selected == "Quản lý Dữ liệu (CRUD)":
    st.title("🛠️ Quản lý Dữ liệu (CRUD)")
    
    # 1. IMPORT
    with st.expander(" Import Dữ liệu mới (CSV)"):
        uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.session_state.df = new_df # Cập nhật vào bộ nhớ
                st.success("Import thành công!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi file: {e}")

    # 2. TÌM KIẾM (SEARCH)
    st.subheader(" Tìm kiếm & Lọc")
    col_search, col_filter = st.columns(2)
    with col_search:
        search_term = st.text_input("Tìm kiếm theo Quận hoặc Loại nhà:")
    with col_filter:
        price_range = st.slider("Khoảng giá (Tỷ)", 0.0, 100.0, (0.0, 100.0))
    
    # Logic lọc
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[
            filtered_df['Quận'].str.contains(search_term, case=False, na=False) | 
            filtered_df['Loại nhà'].str.contains(search_term, case=False, na=False)
        ]
    filtered_df = filtered_df[(filtered_df['Price_Billion'] >= price_range[0]) & (filtered_df['Price_Billion'] <= price_range[1])]

    st.info(f"Hiển thị {len(filtered_df)} bản ghi phù hợp.")

    # 3. HIỂN THỊ & EDIT (UPDATE/DELETE GIÁN TIẾP)
    st.subheader(" Bảng dữ liệu (Cho phép chỉnh sửa)")
    # st.data_editor cho phép sửa trực tiếp trên bảng
    edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True, key="editor")

    # Nút lưu thay đổi
    if st.button(" Lưu thay đổi vào bộ nhớ"):
        # Cập nhật lại session_state (Lưu ý: logic này đơn giản, chỉ cập nhật trên các dòng đang lọc)
        # Trong thực tế cần map theo ID, ở đây ta cập nhật toàn bộ nếu không lọc, hoặc cảnh báo.
        st.session_state.df = edited_df
        st.success("Đã lưu dữ liệu!")
    
    # 4. EXPORT
    st.subheader(" Export Dữ liệu")
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Tải xuống file CSV",
        data=csv,
        file_name='housing_data_export.csv',
        mime='text/csv',
    )

# =========================================================
# MODULE 3: PHÂN TÍCH & TRỰC QUAN HÓA
# =========================================================
elif selected == "Phân tích Trực quan":
    st.title(" Phân tích các yếu tố ảnh hưởng Giá nhà")

    if df.empty:
        st.warning("Không có dữ liệu.")
        st.stop()

    # Tab phân chia các góc nhìn
    tab1, tab2, tab3 = st.tabs([" Vị trí & Giá", " Đặc điểm & Giá", " Tương quan chi tiết"])

    with tab1:
        st.subheader("Phân tích theo Quận/Huyện")
        # Biểu đồ cột: Giá trung bình theo quận
        avg_price_quan = df.groupby('Quận')['Price_Billion'].mean().sort_values(ascending=False).reset_index()
        fig_bar = px.bar(avg_price_quan, x='Quận', y='Price_Billion', color='Price_Billion',
                         title="Giá nhà trung bình theo Quận", labels={'Price_Billion': 'Giá TB (Tỷ)'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Biểu đồ Boxplot: Phân bố giá
        fig_box = px.box(df, x='Quận', y='Price_Billion', color='Quận', title="Phân bố dải giá theo Quận")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.subheader("Phân tích theo Loại hình & Đặc điểm")
        col_a, col_b = st.columns(2)
        with col_a:
            # Pie chart: Tỷ lệ các loại nhà
            type_counts = df['Loại nhà'].value_counts().reset_index()
            type_counts.columns = ['Loại nhà', 'Số lượng']
            fig_pie = px.pie(type_counts, values='Số lượng', names='Loại nhà', title="Tỷ lệ các loại hình BĐS")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            # Scatter: Diện tích vs Giá (phân màu theo loại nhà)
            fig_scatter = px.scatter(df, x='Area_m2', y='Price_Billion', color='Loại nhà', 
                                     size='Floors', hover_data=['Quận'], title="Tương quan Diện tích - Giá")
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("Ma trận tương quan (Correlation)")
        # Chỉ lấy các cột số để tính tương quan
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        # Lọc bớt các cột One-hot để đỡ rối (chỉ lấy các cột chính)
        main_cols = ['Price_Billion', 'Area_m2', 'Bedrooms', 'Bathrooms', 'Floors', 'Facade', 'Dist_Center_km']
        corr_matrix = numeric_df[main_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Mức độ ảnh hưởng giữa các yếu tố")
        st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# MODULE 4: DỰ BÁO GIÁ (PREDICTION)
# =========================================================
elif selected == "Dự báo Giá nhà":
    st.title(" Mô hình Dự đoán Giá trị")
    st.write("Nhập thông số để ước tính giá trị Bất động sản.")

    # Load Model thông minh (Tự sửa lỗi list)
    @st.cache_resource
    def load_model_ai():
        try:
            # 1. Load file model
            loaded_object = joblib.load('model_v2_80percent.pkl')
            
            # 2. Kiểm tra xem file load lên là Model hay là List
            model = None
            if hasattr(loaded_object, 'predict'):
                # Trường hợp chuẩn: File chỉ chứa đúng 1 model
                model = loaded_object
            elif isinstance(loaded_object, list):
                # Trường hợp lỗi của bạn: File chứa 1 danh sách (List)
                # st.warning(f"Phát hiện file chứa danh sách {len(loaded_object)} phần tử. Đang tìm Model...")
                
                # Duyệt qua từng phần tử trong list để tìm cái nào là Model (có hàm predict)
                for item in loaded_object:
                    if hasattr(item, 'predict'):
                        model = item
                        break
            
            # 3. Load danh sách cột
            cols = joblib.load('model_columns_v2.pkl')
            
            return model, cols
            
        except Exception as e:
            st.error(f"Chi tiết lỗi load model: {e}")
            return None, None
    model, model_columns = load_model_ai()

    if model is None:
        st.error("⚠️ Chưa tìm thấy file Model (`model_v2_80percent.pkl`). Vui lòng kiểm tra thư mục.")
        st.stop()

    # Form nhập liệu
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            area = st.number_input("Diện tích (m2)", 20.0, 500.0, 60.0)
            bedroom = st.number_input("Phòng ngủ", 1, 10, 2)
            bathroom = st.number_input("Phòng tắm", 1, 10, 2)
        with c2:
            floors = st.number_input("Số tầng", 1, 10, 1)
            facade = st.number_input("Mặt tiền (m)", 1.0, 20.0, 4.0)
            dist = st.number_input("Cách trung tâm (km)", 0.0, 30.0, 5.0)
        with c3:
            # Lấy danh sách quận/loại từ data thực tế
            quan_list = [c.replace('Dist_', '') for c in model_columns if c.startswith('Dist_')]
            type_list = [c.replace('Type_', '') for c in model_columns if c.startswith('Type_')]
            
            quan_val = st.selectbox("Quận", quan_list)
            type_val = st.selectbox("Loại nhà", type_list)

        st.write("Tiện ích khác:")
        chk_security = st.checkbox("An ninh tốt", True)
        chk_redbook = st.checkbox("Sổ đỏ chính chủ", True)
        
        btn_predict = st.form_submit_button("🚀 ĐỊNH GIÁ NGAY")

    if btn_predict:
        # Chuẩn bị dữ liệu Input khớp với Model Columns
        input_data = pd.DataFrame(index=[0], columns=model_columns).fillna(0)
        
        # Gán giá trị số
        input_data['Area_m2'] = area
        input_data['Bedrooms'] = bedroom
        input_data['Bathrooms'] = bathroom
        input_data['Floors'] = floors
        input_data['Facade'] = facade
        input_data['Dist_Center_km'] = dist
        input_data['Security'] = 1 if chk_security else 0
        input_data['Red_Book'] = 1 if chk_redbook else 0 # Giả định model dùng 0/1 cho sổ đỏ

        # Gán One-hot
        if f'Dist_{quan_val}' in input_data.columns:
            input_data[f'Dist_{quan_val}'] = 1
        if f'Type_{type_val}' in input_data.columns:
            input_data[f'Type_{type_val}'] = 1
            
        # Dự đoán
        try:
            price_pred = model.predict(input_data)[0]
            st.success(f"💰 Giá dự đoán: **{price_pred:,.2f} Tỷ VNĐ**")
        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")