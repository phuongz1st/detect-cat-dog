import streamlit as st
import tensorflow as tf
# Đổi import từ mobilenet_v2 sang resnet50
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# 1. Cấu hình trang (Bắt buộc phải ở dòng đầu tiên)
st.set_page_config(page_title="AI Pet Detector", page_icon="✨", layout="centered")

# 2. Inject Custom CSS (Cập nhật thêm màu Warning cho độ tin cậy thấp)
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 4rem 3rem !important;
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        max-width: 800px;
        margin-top: 3rem;
        margin-bottom: 3rem;
    }

    h1 {
        background: -webkit-linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    .author-text {
        text-align: center;
        font-size: 12px;
        color: #94a3b8;
        margin-top: -1.5rem;
        margin-bottom: 3rem;
    }

    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        background-color: #f8fafc;
        transition: all 0.3s ease;
        padding: 3rem;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #667eea !important;
        background-color: #eff6ff;
        transform: translateY(-2px);
    }

    .stProgress > div > div > div {
        background-color: #667eea;
        border-radius: 10px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 3rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
        width: 100%;
        margin-top: 2rem;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.4);
        color: white;
    }

    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    [data-testid="stImage"] img:hover {
        transform: scale(1.02);
    }

    @keyframes fadeInBounce {
        0% { opacity: 0; transform: scale(0.8) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    .result-badge {
        display: block;
        padding: 20px 30px;
        border-radius: 16px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-top: 3rem;
        margin-bottom: 2rem;
        animation: fadeInBounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        color: white;
    }
    
    .badge-cat { background: linear-gradient(135deg, #f06292 0%, #c2185b 100%); }
    .badge-dog { background: linear-gradient(135deg, #66bb6a 0%, #388e3c 100%); }
    .badge-unknown { background: linear-gradient(135deg, #78909c 0%, #455a64 100%); }
    /* Thêm CSS cho badge cảnh báo độ tin cậy thấp */
    .badge-warning { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }

    .prediction-details-title {
        font-weight: 600;
        font-size: 18px;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .other-predictions-list {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        max-width: 600px;
        margin: 0 auto;
    }
    
    .other-pred-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #475569;
        font-size: 14px;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background-color 0.2s;
    }
    .other-pred-item:hover {
        background-color: #f1f5f9;
    }
    .other-pred-label { font-weight: 500; }
    .other-pred-category { color: #94a3b8; }
    .other-pred-prob { font-weight: 600; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Header UI
st.markdown("<h1>🐾 AI Pet Detector</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Project Web Deploy • Detect Cat & Dog</div>", unsafe_allow_html=True)
st.markdown("<p class='author-text'>By Nguyễn Đông Phương - 2286400025</p>", unsafe_allow_html=True)

# 4. Load AI Model (Đã đổi sang ResNet50)
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet')
    return model

with st.spinner('✨ Initializing ResNet50 Model... This might take a moment.'):
    model = load_model()

# 5. Giao diện Upload
uploaded_file = st.file_uploader("Upload or drag & drop a clear photo of your pet", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert('RGB')
    st.image(image_data, use_container_width=True)
    
    if st.button('✨ Phân Tích Hình Ảnh'):
        # Thanh loading giả lập mượt mà (đã sửa lỗi text)
        progress_text = "✨ Đang phân tích các đặc điểm hình ảnh..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        # Ẩn thanh loading đi cho gọn UI
        time.sleep(0.2)
        my_bar.empty()
        
        # Tiền xử lý cho ResNet50
        img = image_data.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Dự đoán
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]
        
        cat_keywords = ['cat', 'tabby', 'tiger', 'siamese', 'persian', 'lynx', 'leopard', 'kitten', 'cougar', 'lion', 'panther', 'cheetah', 'jaguar']
        dog_keywords = [
            'dog', 'terrier', 'retriever', 'spaniel', 'shepherd', 'hound', 'boxer', 'bulldog', 'dalmatian', 
            'husky', 'corgi', 'pug', 'pomeranian', 'chihuahua', 'beagle', 'collie', 'poodle', 'rottweiler', 
            'doberman', 'shiba', 'akita', 'malamute', 'samoyed', 'chow', 'dane', 'mastiff', 'bernese', 
            'newfoundland', 'schnauzer', 'pinscher', 'sheepdog', 'pointer', 'vizsla', 'setter', 'maltese', 
            'papillon', 'pekingese', 'spitz', 'whippet', 'basenji', 'borzoi', 'greyhound', 'bloodhound', 'wolf'
        ]

        processed_preds = []
        for pred_raw in decoded_preds:
            _, label_raw, prob_raw = pred_raw
            label_clean = label_raw.replace('_', ' ').title()
            
            category = "Other"
            if any(k in label_raw.lower() for k in cat_keywords):
                category = "Cat"
            elif any(k in label_raw.lower() for k in dog_keywords):
                category = "Dog"
            
            processed_preds.append((label_clean, category, prob_raw))

        top_label, top_cat, top_prob = processed_preds[0]
        
        # LOGIC ĐỘ TIN CẬY (Ngưỡng 50%)
        if top_prob < 0.50:
            badge_class = "badge-warning"
            emoji = "🤔"
            display_text = f"Hơi khó đoán... Có thể là {top_cat} ({top_label})"
        else:
            if top_cat == "Cat":
                emoji = "🐱"
                badge_class = "badge-cat"
            elif top_cat == "Dog":
                emoji = "🐶"
                badge_class = "badge-dog"
            else:
                emoji = "❓"
                badge_class = "badge-unknown"
            display_text = f"{top_cat} - {top_label}"

        # Hiển thị Badge kết quả chính
        st.markdown(f'<div class="result-badge {badge_class}">{emoji} {display_text}</div>', unsafe_allow_html=True)
        
        # Hiển thị độ tin cậy
        st.progress(float(top_prob), text=f"Độ tin cậy (Confidence): {top_prob:.2%}")

        # Hiển thị các dự đoán khác
        if len(processed_preds) > 1:
            st.markdown("<p class='prediction-details-title'>Các dự đoán khác có thể xảy ra:</p>", unsafe_allow_html=True)
            st.markdown("<div class='other-predictions-list'>", unsafe_allow_html=True)
            for label, cat, prob in processed_preds[1:]:
                other_emoji = "🐱" if cat == "Cat" else "🐶" if cat == "Dog" else "❓"
                st.markdown(
                    f'<div class="other-pred-item">'
                    f'    <span class="other-pred-label">{other_emoji} {label}</span>'
                    f'    <span class="other-pred-category">({cat})</span>'
                    f'    <span class="other-pred-prob">{prob:.2%}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
