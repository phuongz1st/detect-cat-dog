import streamlit as st
import tensorflow as tf
# Tải ResNet50 để có preprocess_input và decode_predictions đúng cách
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Đặt cấu hình trang
st.set_page_config(page_title="AI Pet Detector", page_icon="🐾", layout="centered")

@st.cache_resource
def load_model():
    # Sử dụng ResNet50 mạnh mẽ hơn cho độ chính xác cao hơn, như ảnh gợi ý
    model = ResNet50(weights='imagenet')
    return model

with st.spinner('✨ Đang khởi tạo mô hình AI ResNet50... xin vui lòng chờ một lát!'):
    model = load_model()

# CSS tùy chỉnh để làm đẹp giao diện và thêm hình nền
custom_css = """
<style>
    /* Hình nền gradient cho toàn bộ trang */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* Container trung tâm (Card) */
    .block-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 4rem 3rem !important;
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        max-width: 800px;
        margin-top: 3rem;
        margin-bottom: 3rem;
    }

    /* Phông chữ chính */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* Tiêu đề chính */
    h1 {
        color: #4c4c9d;
        text-align: center;
        font-weight: 700 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }

    /* Phụ đề */
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Phần thông tin nhóm */
    .team-info {
        text-align: center;
        font-weight: 600;
        color: #555;
        margin-bottom: 3.5rem;
        padding: 1rem;
        background-color: #f1f3f5;
        border-radius: 10px;
    }

    /* Nút dự đoán */
    .stButton > button {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 30px;
        border: none;
        padding: 12px 40px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 117, 252, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 117, 252, 0.4);
        color: white;
    }

    /* Tùy chỉnh Vùng upload ảnh */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #a0a0ff !important;
        border-radius: 15px !important;
        background-color: #f8f9fa;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #2575fc !important;
        background-color: #e6e6ff;
    }

    /* Hiển thị ảnh upload */
    [data-testid="stImage"] img {
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Hiển thị kết quả chính */
    .result-main {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 2.5rem 0;
        padding: 2rem;
        border-radius: 20px;
        animation: fadeInBounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    }
    .result-main.cat {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .result-main.dog {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .result-main.other {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }

    /* Chi tiết Top 3 */
    .result-details-title {
        font-weight: 600;
        font-size: 1.2rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .result-details {
        animation: fadeIn 0.5s;
        max-width: 600px;
        margin: 0 auto;
    }
    .pred-item {
        display: flex;
        justify-content: space-between;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        transition: all 0.2s;
    }
    .pred-item:hover {
        transform: scale(1.02);
        background-color: #f1f3f5;
    }
    .pred-item:nth-child(1) { background-color: #f1f3f5; border: 1px solid #dee2e6; font-weight: bold; font-size: 1.1rem;}

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInBounce {
        0% { opacity: 0; transform: scale(0.8) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Hiển thị tiêu đề
st.markdown("<h1>🐾 AI Pet Detector 🐾</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Project Web Deploy • Detect Cat & Dog using ResNet50</div>", unsafe_allow_html=True)

# Hiển thị thông tin nhóm (Yêu cầu chính)
st.markdown(
    """
    <div class="team-info">
        Người thực hiện:<br>
        Nhóm: Nguyễn Đông Phương, Trần Tuấn Kiệt, trần trọng thành
    </div>
    """,
    unsafe_allow_html=True
)

# Giao diện Upload ảnh
uploaded_file = st.file_uploader("Upload or drag & drop a clear photo of your pet", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh user upload với góc tròn và kênh màu đúng
    image_data = Image.open(uploaded_file).convert('RGB')
    st.image(image_data, use_container_width=True)
    
    # Nút bấm dự đoán
    if st.button('Dự đoán ngay'):
        with st.spinner('Mô hình AI đang phân tích bức ảnh... xin chờ!'):
            # Xử lý ảnh (Resize về 224x224 như ResNet50 yêu cầu)
            img = image_data.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Dự đoán
            preds = model.predict(x)
            decoded_preds = decode_predictions(preds, top=3)[0]
            
            # Logic xác định kết quả chính (Top 1)
            top_class_id, top_class_name, top_prob = decoded_preds[0]
            top_label_lower = top_class_name.lower()
            
            # Làm sạch nhãn (Xóa dấu gạch dưới, viết hoa chữ cái đầu)
            top_label_cleaned = top_class_name.replace('_', ' ').title()

            # Định nghĩa màu sắc thẻ kết quả
            if 'cat' in top_label_lower:
                result_class = "cat"
                emoji = "🐱"
                display_text = f"Kết quả: ĐÂY LÀ MÈO"
            elif 'dog' in top_label_lower:
                result_class = "dog"
                emoji = "🐶"
                display_text = f"Kết quả: ĐÂY LÀ CHÓ"
            else:
                result_class = "other"
                emoji = "❓"
                display_text = f"Kết quả: KHÔNG RÕ LÀ CHÓ HAY MÈO"
            
            # Hiển thị kết quả chính với màu sắc và hoạt ảnh
            st.markdown(
                f'<div class="result-main {result_class}">'
                f'{emoji} {display_text} <br> '
                f'<i>({top_label_cleaned})</i>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Hiển thị Top 3 chi tiết
            st.markdown("<p class='result-details-title'>Chi tiết các dự đoán hàng đầu:</p>", unsafe_allow_html=True)
            
            # Sử dụng progress bar đúng cách cho Top 1 để minh họa, Top 2-3 trong danh sách
            # Sửa lỗi: Chỉ sử dụng văn bản thuần túy trong `text` của progress
            st.progress(float(top_prob), text=f"✨ {top_label_cleaned}: {top_prob:.2%} (Độ tin cậy hàng đầu)")
            
            # Danh sách Top 3 sạch hơn
            st.markdown("<div class='result-details'>", unsafe_allow_html=True)
            for _, class_name, prob in decoded_preds:
                label_cleaned = class_name.replace('_', ' ').title()
                label_lower = class_name.lower()
                
                if 'cat' in label_lower: pred_emoji = "🐱"
                elif 'dog' in label_lower: pred_emoji = "🐶"
                else: pred_emoji = "🐾"

                st.markdown(
                    f'<div class="pred-item">'
                    f'<span>{pred_emoji} {label_cleaned}</span>'
                    f'<span>{prob:.2%}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #aaa; font-size: 0.9rem;'>Phân tích hình ảnh được cung cấp bởi ResNet50 Pretrained Model trên tập dữ liệu ImageNet.</p>", unsafe_allow_html=True)
