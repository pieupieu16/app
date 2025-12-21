import pandas as pd
import matplotlib.pyplot as plt
import shap
import io
import numpy as np

# 1. Hàm vẽ biểu đồ SHAP (Đã tối ưu)
def plot_shap_waterfall(model, input_data, model_columns=None):
    try:
        # Tách mô hình từ Pipeline nếu có
        regressor = model.steps[-1][1] if hasattr(model, 'named_steps') else model
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer(input_data, check_additivity=False)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        plt.yticks(fontsize=8)
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        plt.tight_layout()

        # Lưu vào bộ nhớ đệm (Buffer)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf
    except Exception as e:
        return f"Error: {str(e)}"

# 2. Hàm tính toán khu vực rẻ nhất (Cheapest District)
def get_cheapest_district(df, col_price, col_area):
    if df.empty or col_price not in df.columns or col_area not in df.columns:
        return "N/A"
    
    # Tính giá trung bình trên mỗi m2
    temp_df = df[df[col_area] > 0].copy()
    temp_df['price_per_m2'] = temp_df[col_price] / temp_df[col_area]
    
    # Giả sử bạn đã có cột 'Quận' sau khi xử lý One-Hot
    if 'Quận' in temp_df.columns:
        result = temp_df.groupby('Quận')['price_per_m2'].mean().idxmin()
        return result
    return "N/A"


def get_data_summary(df):
    if df.empty:
        return "Không có dữ liệu."
    
    # Tính toán các thông số chính (Key Metrics)
    avg_price = df['Giá nhà'].mean()
    top_districts = df.groupby('Quận')['Giá nhà'].mean().nlargest(3).to_dict()
    
    summary = f"""
    Tóm tắt dữ liệu thực tế trên biểu đồ:
    - Giá trung bình toàn thành phố: {avg_price:.2f} Tỷ.
    - 3 Quận có giá trung bình cao nhất: {top_districts}.
    - Tổng số bản ghi (Total records): {len(df)} căn nhà.
    """
    return summary