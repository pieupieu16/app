
import shap
import matplotlib.pyplot as plt
import numpy as np

def plot_shap_waterfall(model, input_data, model_columns):
    """
    Hàm vẽ biểu đồ Waterfall cho Sklearn Pipeline
    """
    try:
        # BƯỚC 1: BÓC TÁCH PIPELINE
        # Pipeline của bạn có dạng: Preprocessor -> Imputer -> Model
        # Chúng ta cần lấy 'model' ra để giải thích, và dùng các bước trước đó để transform dữ liệu
        
        # 1.1. Lấy mô hình lõi (GradientBoostingRegressor)
        # 'model' là tên step cuối cùng bạn đặt trong Pipeline
        regressor = model.named_steps['model']
        
        # 1.2. Chạy dữ liệu qua các bước tiền xử lý (Preprocessor & Imputer)
        # input_data đang là DataFrame thô, cần biến đổi thành dạng số (numpy array) mà model hiểu
        data_transformed = input_data.copy()
        
        # Duyệt qua các bước TRỪ bước cuối cùng (model)
        for name, step in model.named_steps.items():
            if name != 'model': 
                data_transformed = step.transform(data_transformed)
        
        # BƯỚC 2: TẠO EXPLAINER VỚI MÔ HÌNH LÕI
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer(data_transformed)
        
        # BƯỚC 3: TÁI TẠO TÊN CỘT (QUAN TRỌNG)
        # ColumnTransformer sẽ đảo thứ tự cột (Num trước, Cat sau), nên ta cần lấy lại tên đúng
        feature_names = []
        try:
            # Cách chuẩn cho Scikit-learn phiên bản mới
            preprocessor = model.named_steps['preprocessor']
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                # Fallback: Tự dựng lại tên cột dựa trên cấu hình transformer
                # Lấy danh sách cột nhóm 'num' (RobustScaler)
                num_cols = preprocessor.transformers_[0][2] 
                # Lấy danh sách cột nhóm 'cat' (passthrough)
                cat_cols = preprocessor.transformers_[1][2]
                feature_names = list(num_cols) + list(cat_cols)
        except:
            # Nếu mọi cách đều lỗi thì đặt tên chung chung
            feature_names = [f"Feature {i}" for i in range(data_transformed.shape[1])]

        # Gán tên cột vào đối tượng SHAP để hiển thị lên biểu đồ
        shap_values.feature_names = feature_names

        # BƯỚC 4: VẼ BIỂU ĐỒ
        fig, ax = plt.subplots(figsize=(12, 8)) # Tăng kích thước để dễ đọc
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        
        # Thêm tiêu đề
        current_pred = shap_values[0].values.sum() + explainer.expected_value[0]
        plt.title(f"Phân tích giá: {current_pred:.2f} Tỷ (Base: {explainer.expected_value[0]:.2f})", fontsize=16)
        plt.tight_layout()
        
        return fig

    except Exception as e:
        return f"Không thể tạo giải thích: {str(e)}"