import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
import io
# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hanoi Real Estate Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; border: 1px solid #e6e6e6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

# --- 1. XỬ LÝ DỮ LIỆU (BACKEND) ---
# Đổi tên hàm thành 'load_data_v2' để bắt buộc Streamlit xóa cache cũ
@st.cache_data
def load_data_v2():
    file_path = 'dự tính giá nhà - Trang tính1 (2).csv'
    try:
        # Đọc file
        df = pd.read_csv(file_path) # Pandas tự động detect encoding tốt, nhưng có thể thử encoding='utf-8-sig' nếu lỗi font
        
        # 1. Làm sạch tên cột (Xóa khoảng trắng thừa)
        df.columns = df.columns.str.strip()
        
        # 2. Đổi tên cột (Mapping)
        rename_mapping = {
            'Giá(ty)': 'Giá (Tỷ)',
            'Diện Tích(m2)': 'Diện tích (m2)',
            'numberbedroom': 'Phòng ngủ',
            'numberbathroom': 'Phòng tắm',
            'Loại Hình(căn hộ ,nhà,villa)': 'Loại nhà',
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
        # ... (Phần 3. Làm sạch dữ liệu)

        # Ép kiểu cột Giá và Diện tích về số (nếu có chữ sẽ biến thành NaN)
        
        # 🟢 THÊM 3 DÒNG CODE MỚI NÀY ĐỂ XỬ LÝ KÝ TỰ RÁC 🟢
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].astype(str).str.strip() # 1. Loại bỏ khoảng trắng đầu/cuối
        # 2. Loại bỏ các ký tự phổ biến gây lỗi (ví dụ: 'tỷ', 'ty' hoặc dấu cách giữa số)
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].str.replace('tỷ', '', regex=False).str.replace('ty', '', regex=False).str.replace(' ', '', regex=False)
        # 3. Loại bỏ ký tự không phải số hoặc dấu chấm thập phân (ví dụ: #, %, v.v.)
        df['Giá (Tỷ)'] = df['Giá (Tỷ)'].str.replace(r'[^\d.]', '', regex=True) 
        
        # Sau đó mới gọi hàm chuyển đổi
        df['Giá (Tỷ)'] = pd.to_numeric(df['Giá (Tỷ)'], errors='coerce')
        df['Diện tích (m2)'] = pd.to_numeric(df['Diện tích (m2)'], errors='coerce')
        
        # ... (Phần code tiếp theo)

        # 4. GỘP CỘT QUẬN (QUAN TRỌNG)
        quan_columns = ['Ba Đình', 'Cầu Giấy', 'Đống Đa', 'Hai Bà Trưng', 'Thanh Xuân', 
                        'Hoàng Mai', 'Long Biên', 'Hà Đông', 'Tây Hồ', 'Nam Từ Liêm', 
                        'Bắc Từ Liêm', 'Thanh Trì']
        
        # Tìm các cột quận thực tế có trong file
        valid_quan_cols = [q for q in quan_columns if q in df.columns]

        if not valid_quan_cols:
            # Nếu không tìm thấy cột quận nào, tạo cột mặc định
            df['Quận'] = "Chưa xác định"
        else:
            # Hàm xác định quận cho từng dòng
            def get_quan(row):
                for q in valid_quan_cols:
                    if row.get(q) == 1.0:
                        return q
                return "Khác"
            
            df['Quận'] = df.apply(get_quan, axis=1)

        # 5. Tổng tiện ích
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

# SIDEBAR
with st.sidebar:
    st.title("🏢 Hanoi Housing Hub")
    
    # Nút Reset mạnh tay hơn
    if st.button("⚠️ Reset toàn bộ Ứng dụng"):
        st.cache_data.clear()
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    menu = st.radio(
        "Điều hướng:",
        ["Dashboard Tổng quan", "Quản lý Dữ liệu (CRUD)", "Làm sạch & Chuẩn hóa", "Phân tích Chuyên sâu", "Tableau Integration"]
    )

# KIỂM TRA AN TOÀN
if df.empty:
    st.warning("Chưa có dữ liệu. Vui lòng kiểm tra file CSV.")
    st.stop()

# --- MODULE 1: DASHBOARD TỔNG QUAN ---
if menu == "Dashboard Tổng quan":
    st.title("📊 Dashboard Tổng quan Thị trường")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số tin đăng", f"{len(df):,}")
    col2.metric("Giá trung bình", f"{df['Giá (Tỷ)'].mean():.2f} Tỷ")
    col3.metric("Diện tích trung bình", f"{df['Diện tích (m2)'].mean():.1f} m²")
    try:
        top_quan = df['Quận'].mode()[0]
    except:
        top_quan = "N/A"
    col4.metric("Khu vực sôi động nhất", top_quan)

    # CSS tùy chỉnh
    st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; border: 1px solid #e6e6e6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);}
    
    /* 🎨 Sửa màu tại đây */
    .stMetricLabel {color: #6c757d !important;} /* Đổi thành Xám đậm */
    .stMetricValue {color: #007bff !important;} /* Đổi thành Xanh lam đậm */
    
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Phân bổ Giá theo Diện tích")
        fig_map = px.scatter(df, x="Diện tích (m2)", y="Giá (Tỷ)", color="Quận", size="Giá (Tỷ)")
        st.plotly_chart(fig_map, use_container_width=True)
    
    with c2:
        st.subheader("Tỷ lệ Loại hình nhà")
        df['Loại nhà'] = df['Loại nhà'].astype(str).str.strip()
        fig_pie = px.pie(df, names='Loại nhà', title='Cơ cấu nguồn cung', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- MODULE 2: QUẢN LÝ DỮ LIỆU (CRUD) ---
elif menu == "Quản lý Dữ liệu (CRUD)":
    st.title("📂 Trung tâm Dữ liệu (Data Center)")
    
    # Import
    with st.expander("Nhập dữ liệu mới (Import)"):
        uploaded_file = st.file_uploader("Tải lên file CSV/Excel khác", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)
            st.session_state['data'] = new_df
            st.success("Tải dữ liệu mới thành công!")
            st.rerun()

    # CRUD Check
    st.subheader("Xem & Chỉnh sửa Dữ liệu")
    
    # Bộ lọc an toàn
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        unique_quan = list(df['Quận'].unique())
        search_district = st.multiselect("Lọc theo Quận:", unique_quan, default=unique_quan[:3] if len(unique_quan)>0 else None)
    with filter_col2:
        max_p = float(df['Giá (Tỷ)'].max()) if not df.empty else 100.0
        price_range = st.slider("Khoảng giá (Tỷ):", 0.0, max_p, (0.0, max_p))
    
    df_display = df.copy()
    if search_district:
        df_display = df_display[df_display['Quận'].isin(search_district)]
    df_display = df_display[(df_display['Giá (Tỷ)'] >= price_range[0]) & (df_display['Giá (Tỷ)'] <= price_range[1])]

    edited_df = st.data_editor(df_display, num_rows="dynamic", use_container_width=True)
    
    if st.button("Lưu thay đổi tạm thời"):
        st.session_state['data'] = edited_df
        st.success("Đã cập nhật!")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Tải xuống CSV", csv, "data_final.csv", "text/csv")
    # 3. EXPORT
    st.subheader("3. Export Dữ liệu")
    
    # ----------------------------------------------------
    # LOGIC XUẤT RA XLSX VÀ LÀM TRÒN SỐ (Không đổi dấu thập phân)
    # ----------------------------------------------------
    # 1. Tạo bản sao để không thay đổi dữ liệu gốc
    df_export = df.copy()
    
    # 2. Định danh các cột số cần làm tròn
    numeric_cols_for_export = ['Giá (Tỷ)', 'Diện tích (m2)', 'Phòng ngủ', 'Phòng tắm'] 
    
    # 3. Làm tròn và đảm bảo định dạng số
    for col in numeric_cols_for_export:
        # Ép kiểu lại thành số, làm tròn 2 chữ số thập phân
        df_export[col] = pd.to_numeric(df_export[col], errors='coerce').round(2) 

    # 4. Sử dụng BytesIO để tạo buffer Excel (.xlsx)
    buffer = io.BytesIO()
    
    try:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Data Sạch')
        
        buffer.seek(0)
        
        # Nút Download Excel (.xlsx)
        st.download_button(
            label="Tải xuống XLSX (Đã làm tròn)",
            data=buffer,
            file_name="data_cleaned_rounded.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except ImportError:
        st.error("Lỗi: Vui lòng cài đặt thư viện 'openpyxl' bằng lệnh 'pip install openpyxl'")

    # Vẫn giữ nút CSV cũ (cho Tableau)
    csv_string = df.to_csv(index=False, encoding='utf-8-sig') 
    st.download_button(
        "Tải xuống CSV (Cho Tableau)", 
        csv_string.encode('utf-8-sig'), 
        "data_cleaned_for_tableau.csv", 
        "text/csv"
    )

# --- MODULE 3: LÀM SẠCH & CHUẨN HÓA ---
elif menu == "Làm sạch & Chuẩn hóa":
    st.title("✨ Data Refinery (Làm sạch)")
    
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
            
        threshold = st.number_input("Giá trần lọc ngoại lai (Tỷ):", value=500)
        if st.button("Loại bỏ ngoại lai"):
            df = df[df['Giá (Tỷ)'] <= threshold]
            st.session_state['data'] = df
            st.warning("Đã loại bỏ ngoại lai!")
            st.rerun()

# --- MODULE 4: PHÂN TÍCH CHUYÊN SÂU ---
elif menu == "Phân tích Chuyên sâu":
    st.title("📈 Phân tích & Trực quan hóa")
    
    st.subheader("1. Tương quan: Giá & Diện tích")
    fig1 = px.scatter(df, x="Diện tích (m2)", y="Giá (Tỷ)", color="Quận", 
                      size="Tổng tiện ích", trendline="ols")
    st.plotly_chart(fig1, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("2. Top Quận đắt đỏ nhất")
        avg_price = df.groupby('Quận')['Giá (Tỷ)'].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(avg_price, x='Quận', y='Giá (Tỷ)', color='Giá (Tỷ)')
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("3. Phân phối giá theo Loại nhà")
        fig3 = px.box(df, x="Loại nhà", y="Giá (Tỷ)", color="Loại nhà") 
        st.plotly_chart(fig3, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("4. Giá theo Số phòng ngủ")
        if 'Phòng ngủ' in df.columns:
            df_bed = df[df['Phòng ngủ'] <= 10]
            bed_trend = df_bed.groupby('Phòng ngủ')['Giá (Tỷ)'].mean().reset_index()
            fig4 = px.line(bed_trend, x='Phòng ngủ', y='Giá (Tỷ)', markers=True)
            st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        st.subheader("5. Cấu trúc thị trường")
        df_tree = df[df['Quận'] != 'Khác']
        fig5 = px.treemap(df_tree, path=['Quận', 'Loại nhà'], values='Giá (Tỷ)')
        st.plotly_chart(fig5, use_container_width=True)

# --- MODULE 5: TABLEAU INTEGRATION ---
elif menu == "Tableau Integration":
    st.title("🌐 Kết nối Tableau")
    st.markdown("""
    Đây là khu vực tích hợp Dashboard từ Tableau Public. 
    Bạn có thể tương tác (Lọc, Zoom, Click) trực tiếp ngay tại đây.
    """)
    
    # --- CÁCH LẤY CODE NHÚNG: ---
    # 1. Upload file Tableau của bạn lên Tableau Public (https://public.tableau.com)
    # 2. Mở Dashboard trên web, bấm nút "Share" (Chia sẻ) -> Copy "Embed Code"
    # 3. Dán đoạn code đó vào biến html_code bên dưới.
    
    # Dưới đây là Code mẫu (Demo Dashboard Bất động sản):
    tableau_html_code = """
    <div class='tableauPlaceholder' id='viz1763127239393' style='position: relative'><noscript><a href='#'><img alt='tk ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Book7_17631271401140&#47;tk' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Bo&#47;Book7_17631271401140&#47;tk&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1763127239393');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
    
    # Hiển thị khung Tableau
    components.html(tableau_html_code, height=850, scrolling=True)
    