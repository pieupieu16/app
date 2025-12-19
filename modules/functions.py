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