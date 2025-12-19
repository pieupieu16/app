import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from datetime import datetime

import io
import preprocess
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import re
import base64
from modules.constants import districts, wards_map
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
def load_data(file_path='processed_housing_data.parquet'):
    try:
        df = pd.read_parquet(file_path)
        
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
    options=["Trang chủ", "Quản lý Dữ liệu (CRUD)", "Phân tích Trực quan", "Bản đồ quy hoạch Hà Nội"],
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
# MODULE 1: TRANG CHỦ
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
        c2.metric("Giá trung bình", f"{avg_price:,.2f} Tỷ")
        c3.metric("Khu vực rẻ nhất (m²)", f"{cheapest_district}")
        c4.metric("Căn đắt nhất", f"{max_price:,.2f} Tỷ")
    else:
        st.info("Vui lòng Import dữ liệu ở tab 'Quản lý Dữ liệu' để xem thống kê.")




    def clean_feature_names(names):
        """
        Hàm rút gọn tên và tạo khoảng cách an toàn (Padding).
        """
        cleaned_names = []
        for name in names:
            # 1. Rút gọn từ khóa
            new_name = str(name)
            new_name = new_name.replace("Huyện_Phường", "P.")
            new_name = new_name.replace("Quận_Quận", "Q.")
            new_name = new_name.replace("Tỉnh_Thành phố", "TP.")
            new_name = new_name.replace("Giấy tờ pháp lý", "Pháp lý")
            new_name = new_name.replace("Unknown", "?") # Rút gọn Unknown
            
            # 2. Cắt bớt nếu vẫn quá dài (trên 20 ký tự)
            if len(new_name) > 20:
                new_name = new_name[:18] + ".."
                
            # 3. [QUAN TRỌNG] Thêm khoảng trắng vào cuối
            # Mẹo này giúp đẩy chữ sang trái, tránh bị số liệu đè lên
            new_name = new_name + "      "  # Thêm 6 khoảng trắng
                
            cleaned_names.append(new_name)
        return cleaned_names

    def plot_shap_waterfall(model, input_data, model_columns=None):
        """
        Phiên bản Fix lỗi đè chữ bằng cách tăng padding và kích thước biểu đồ.
        """
        try:
            # --- BƯỚC 1: XỬ LÝ DỮ LIỆU ---
            if isinstance(input_data, pd.Series):
                input_data = input_data.to_frame().T
            
            is_pipeline = hasattr(model, 'named_steps')
            
            # Mặc định lấy tên cột
            raw_feature_names = list(input_data.columns) if hasattr(input_data, 'columns') else [f"F{i}" for i in range(input_data.shape[1])]
            data_transformed = input_data

            if is_pipeline:
                # Pipeline: Transform và lấy feature names
                regressor = model.steps[-1][1] 
                preprocessor = model.steps[0][1]
                try:
                    data_transformed = preprocessor.transform(input_data)
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        raw_feature_names = preprocessor.get_feature_names_out().tolist()
                except:
                    pass
            else:
                # Standalone Model
                regressor = model
                if hasattr(regressor, 'feature_names_in_') and hasattr(input_data, 'columns'):
                    required_cols = regressor.feature_names_in_
                    valid_cols = [c for c in required_cols if c in input_data.columns]
                    if len(valid_cols) == len(required_cols):
                        data_transformed = input_data[required_cols]
                        raw_feature_names = list(required_cols)

            # --- BƯỚC 2: LÀM SẠCH TÊN CỘT ---
            # Gọi hàm clean đã thêm khoảng trắng đệm
            short_feature_names = clean_feature_names(raw_feature_names)

            # --- BƯỚC 3: TÍNH SHAP ---
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer(data_transformed, check_additivity=False)

            # Gán tên cột (xử lý lệch số lượng nếu có)
            if len(short_feature_names) == shap_values.shape[1]:
                shap_values.feature_names = short_feature_names
            elif len(short_feature_names) > shap_values.shape[1]:
                shap_values.feature_names = short_feature_names[:shap_values.shape[1]]
            
            # --- BƯỚC 4: VẼ BIỂU ĐỒ ---
            # Thay vì (50, 15), hãy dùng (12, 8) và thêm dpi=200
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            
            base_val = explainer.expected_value
            if isinstance(base_val, (np.ndarray, list)): base_val = base_val[0]
            current_pred = shap_values[0].values.sum() + base_val
            
            # max_display=14: Hiển thị vừa đủ
            shap.plots.waterfall(shap_values[0], max_display=16, show=False)
            
            # Tùy chỉnh font chữ trục Y nhỏ lại một chút
            plt.yticks(fontsize=11)
            
            plt.title(f"Dự báo: {current_pred:,.0f} (Base: {base_val:,.0f})", fontsize=16)
            plt.tight_layout()
            
            return fig

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Lỗi hiển thị: {str(e)}"
    
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
        # input_data['Năm'] = nam_gd
        # input_data['Tháng'] = thang_gd

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
                        <p style="color: #718096;">~ {(predicted_price*1000000000 / (dien_tich)):,.0f} VNĐ / m²</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with res_c2:
                    st.info("Chi tiết đầu vào")
                    st.write(f"**Vị trí:** {selected_district}")
                    if is_ward_specific:
                        st.write(f"**Phường:** {selected_ward}")
                    st.write(f"**Diện tích:** {dien_tich} m²")
                    st.write(f"**Kết cấu:** {so_tang} tầng, {so_phong} PN")
                # ... (Phần code hiển thị giá dự đoán cũ của bạn) ...

                st.markdown("---")
                st.subheader("🤖 AI Giải thích: Tại sao có mức giá này?")
                
                # Gọi hàm giải thích
                with st.spinner("Đang phân tích các yếu tố tác động..."):
                    # Lưu ý: input_data phải đúng format model yêu cầu (DataFrame)
                    fig_explanation = plot_shap_waterfall(model, input_data, model_columns)
                    
                    if isinstance(fig_explanation, str): # Nếu trả về chuỗi lỗi
                        st.warning(fig_explanation)
                    else:
                        # Chia cột để hiển thị đẹp hơn
                        exp_c1, exp_c2 = st.columns([2, 1])
                        
                        with exp_c1:
                            # Hiển thị biểu đồ
                            st.pyplot(fig_explanation)
                        
                        with exp_c2:
                            st.info("""
                            **Hướng dẫn đọc biểu đồ:**
                            - **Màu Đỏ (+):** Các yếu tố làm TĂNG giá nhà.
                            - **Màu Xanh (-):** Các yếu tố làm GIẢM giá nhà.
                            - **Độ dài:** Mức độ ảnh hưởng (càng dài càng quan trọng).
                            """)

            except Exception as e:
                st.error(f"Lỗi khi dự báo: {str(e)}")
                st.dataframe(input_data) # Debug

## =========================================================
# MODULE 2: QUẢN LÝ DỮ LIỆU (ĐÃ TỐI ƯU HÓA)
# =========================================================
elif selected == "Quản lý Dữ liệu (CRUD)":
    st.title("Quản lý Dữ liệu")

    # --- 1. CẬP NHẬT DỮ LIỆU MỚI ---
    st.subheader("1. Cập nhật dữ liệu mới")
    
    with st.expander("Thêm dữ liệu thô & Chạy Tiền xử lý"):
        st.info("Upload file dữ liệu thô (Raw CSV/Excel). Hệ thống sẽ tự động làm sạch và gộp vào dữ liệu chính.")
        
        # Widget Upload file
        uploaded_raw_file = st.file_uploader("Chọn file dữ liệu thô", type=['csv', 'xlsx'])
        
        # Tùy chọn chế độ gộp
        merge_mode = st.radio(
            "Phương thức cập nhật:",
            options=["Gộp thêm vào dữ liệu cũ (Append)", "Thay thế hoàn toàn (Replace)"],
            horizontal=True
        )
        mode_key = 'append' if "Gộp" in merge_mode else 'replace'
        
        # Nút bấm xử lý
        if uploaded_raw_file is not None:
            if st.button("🚀 Bắt đầu Xử lý & Cập nhật", type="primary"):
                try:
                    with st.spinner("Đang chạy script tiền xử lý (cleaning, mapping, encoding)..."):
                        # A. Đọc file upload
                        if uploaded_raw_file.name.endswith('.csv'):
                            raw_df = pd.read_csv(uploaded_raw_file)
                        else:
                            raw_df = pd.read_excel(uploaded_raw_file)
                        
                        # B. Gọi hàm xử lý (Giả sử bạn có module preprocess)
                        # Lưu ý: Đảm bảo preprocess.run_pipeline trả về DataFrame chuẩn
                        new_final_df = preprocess.run_pipeline(
                            raw_df, 
                            current_df=st.session_state.get('df', pd.DataFrame()), 
                            mode=mode_key
                        )
                        
                        # C. Lưu xuống đĩa
                        new_final_df.to_parquet('processed_housing_data.parquet', index=False)
                        
                        # D. QUAN TRỌNG: Xóa Cache cũ và Cập nhật Session
                        st.cache_data.clear()  # <--- Xóa cache để lần sau load lại dữ liệu mới
                        st.session_state.df = new_final_df
                        
                        st.success(f"✅ Thành công! Tổng số dòng hiện tại: {len(new_final_df)}")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra: {e}")

    # --- 2. XUẤT DỮ LIỆU ---
    st.subheader("2. Xuất dữ liệu ra file")
    
    # Lấy df từ session state
    df = st.session_state.get('df', None)

    if df is not None and not df.empty:
        col1, col2 = st.columns(2)
        
        # --- Xuất CSV (Nhanh, khuyến khích dùng) ---
        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        with col1:
            st.download_button(
                label="📥 Tải xuống CSV (Nhanh)",
                data=csv_data,
                file_name='du_lieu_nha_dat.csv',
                mime='text/csv'
            )
            
        # --- Xuất Excel (Chậm, cần tối ưu) ---
        # Chỉ xử lý Excel nếu dữ liệu < 100.000 dòng (tránh crash)
        with col2:
            # Dùng buffer để không tốn ổ cứng server
            buffer = io.BytesIO()
            
            # Kiểm tra kích thước dữ liệu
            if len(df) > 5000:
                st.warning("Dữ liệu lớn (>5000 dòng). File Excel sẽ không được căn chỉnh cột tự động để đảm bảo tốc độ.")
                is_large_file = True
            else:
                is_large_file = False

            # Nút download trigger việc tạo file
            if st.button("Chuẩn bị file Excel"):
                with st.spinner("Đang tạo file Excel..."):
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                        
                        # Chỉ căn chỉnh cột (Auto-adjust) nếu file nhỏ
                        if not is_large_file:
                            worksheet = writer.sheets['Data']
                            for i, col in enumerate(df.columns):
                                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                                worksheet.set_column(i, i, max_len)
                    
                    buffer.seek(0)
                    st.download_button(
                        label="Click để tải Excel ngay",
                        data=buffer,
                        file_name='du_lieu_nha_dat.xlsx',
                        mime='application/vnd.ms-excel'
                    )
    else:
        st.warning("⚠️ Chưa có dữ liệu nào để xuất.")

    # --- 3. TÌM KIẾM & LỌC (ĐÃ TỐI ƯU HIỂN THỊ) ---
    st.subheader("3. Tìm kiếm & Lọc nhanh")
    
    if df is not None and not df.empty:
        col_search, col_filter = st.columns(2)
        
        with col_search:
            search_term = st.text_input("🔍 Tìm kiếm (Quận/Loại nhà):")
        
        with col_filter:
            # Xử lý an toàn nếu cột giá không tồn tại hoặc toàn NaN
            if COL_PRICE in df.columns and df[COL_PRICE].notna().any():
                max_price = float(df[COL_PRICE].max())
                price_range = st.slider("Khoảng giá (Tỷ)", 0.0, max_price, (0.0, max_price))
            else:
                st.warning("Không tìm thấy cột giá để lọc.")
                price_range = (0, 0)
        
        # Logic lọc dữ liệu
        filtered_df = df.copy()
        
        # 1. Lọc theo giá
        if COL_PRICE in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df[COL_PRICE] >= price_range[0]) & 
                (filtered_df[COL_PRICE] <= price_range[1])
            ]
        
        # 2. Lọc theo từ khóa (Vectorized - Nhanh hơn)
        if search_term:
            # Chuyển về chữ thường để tìm không phân biệt hoa thường
            term = search_term.lower()
            mask = pd.Series(False, index=filtered_df.index)
            
            if COL_DISTRICT in filtered_df.columns:
                mask |= filtered_df[COL_DISTRICT].astype(str).str.lower().str.contains(term, na=False)
            if COL_TYPE in filtered_df.columns:
                mask |= filtered_df[COL_TYPE].astype(str).str.lower().str.contains(term, na=False)
            
            filtered_df = filtered_df[mask]

        st.info(f"📊 Tìm thấy **{len(filtered_df)}** bản ghi phù hợp.")

        # --- HIỂN THỊ DỮ LIỆU THÔNG MINH ---
        # Chỉ cho phép edit trên 1000 dòng đầu để tránh treo trình duyệt
        MAX_ROWS_DISPLAY = 1000
        
        if len(filtered_df) > MAX_ROWS_DISPLAY:
            display_df = filtered_df.head(MAX_ROWS_DISPLAY)
        else:
            display_df = filtered_df

        edited_df = st.data_editor(
            display_df, 
            num_rows="dynamic", 
            use_container_width=True,
            key="data_editor_crud" # Key cố định để tránh render lại không cần thiết
        )

        if st.button("💾 Lưu thay đổi bảng"):
            # Cập nhật lại vào dữ liệu gốc trong session state
            # Lưu ý: Logic này chỉ cập nhật các dòng đang hiển thị
            # Cần xử lý kỹ hơn nếu muốn update ngược lại tập dữ liệu 80k dòng
            st.session_state.df.update(edited_df)
            st.success("Đã lưu dữ liệu vào bộ nhớ tạm!")
            
    else:
        st.warning("Dữ liệu trống.")
# =========================================================
# MODULE 3: PHÂN TÍCH TRỰC QUAN
# =========================================================
elif selected == "Phân tích Trực quan":
    st.title(" Phân tích Giá trị BĐS")

    

    tab1, tab2, tab3,tab4 = st.tabs([" Vị trí & Giá", " Đặc điểm & Giá", "Phân phối giá nhà","Phân tích outline theo khu vực"])

    with tab1:
        st.subheader("Giá trung bình theo Quận")
        tableau_code = """
        <div class='tableauPlaceholder' id='viz1765358854926' style='position: relative'><noscript><a href='#'><img alt='Dashboard 3 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='trcquanhadliuginh&#47;Dashboard3' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1765358854926');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                                                vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        """
        components.html(tableau_code, height=850, scrolling=True)

    with tab2:
        st.subheader("Phân tích theo Loại hình")
        col_a, col_b = st.columns(2)
        with col_a:
            tableau_code = """
            <div class='tableauPlaceholder' id='viz1765358659690' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='trcquanhadliuginh&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1765358659690');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='420px';vizElement.style.maxWidth='650px';vizElement.style.width='100%';vizElement.style.minHeight='587px';vizElement.style.maxHeight='887px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                   
              vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
            """
            components.html(tableau_code, height=850, scrolling=True)
        with col_b:
            tableau_code = """
            <<div class='tableauPlaceholder' id='viz1765359243747' style='position: relative'><noscript><a href='#'><img alt='Dashboard 4 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard4&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='trcquanhadliuginh&#47;Dashboard4' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard4&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1765359243747');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                                                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
            """
            components.html(tableau_code, height=850, scrolling=True)

    with tab3:
        st.subheader("Phân phối giá nhà")
        tableau_code = """
        <div class='tableauPlaceholder' id='viz1765359115044' style='position: relative'><noscript><a href='#'><img alt='Dashboard 5 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard5&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='trcquanhadliuginh&#47;Dashboard5' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard5&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1765359115044');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                                                vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        """
        components.html(tableau_code, height=850, scrolling=True)
        
    with tab4:
        st.subheader("Phân tích outline theo khu vực")
        tableau_code = """
        <div class='tableauPlaceholder' id='viz1765359797054' style='position: relative'><noscript><a href='#'><img alt='Dashboard 2 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='trcquanhadliuginh&#47;Dashboard2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;tr&#47;trcquanhadliuginh&#47;Dashboard2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1765359797054');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                                                vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        """
        components.html(tableau_code, height=850, scrolling=True)
# =========================================================
# MODULE 4:
# =========================================================
elif selected == "Bản đồ quy hoạch Hà Nội":
    # 1. Hàm chuyển ảnh sang Base64
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # 2. Sử dụng trong giao diện
    st.subheader("Tra cứu Quy hoạch")

    img_file = 'Ảnh chụp màn hình 2025-12-10 223420.png'  # Tên file ảnh của bạn
    target_url = 'https://quyhoach.hanoi.vn'

    try:

        img_base64 = get_base64_of_bin_file(img_file)
        st.markdown(
            f"""
            <a href="{target_url}" target="_blank">
                <img src="data:image/jpeg;base64,{img_base64}" width="100%" style="border-radius: 5px;">(Nhấn vào ảnh để xem chi tiết)</p>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Không tìm thấy file ảnh bản đồ.")
    

    