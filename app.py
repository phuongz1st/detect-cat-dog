import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Cấu hình trang Web
st.set_page_config(page_title="Cat vs Dog Detector", page_icon="🐾")

st.title(" Cat vs Dog Detector ")
st.write("Project Web Deploy - Detect Cat & Dog")
st.write("Nhóm 6")

# 2. Load Model
@st.cache_resource
def load_model():
    # Sử dụng MobileNetV2 đã train sẵn trên ImageNet (nhanh, nhẹ, chính xác cao)
    model = MobileNetV2(weights='imagenet')
    return model

with st.spinner('Đang tải model... vui lòng chờ chút nhé!'):
    model = load_model()

# 3. Giao diện Upload ảnh
uploaded_file = st.file_uploader("Chọn ảnh chó hoặc mèo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh user upload
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Ảnh đã upload', use_container_width=True)
    
    # Nút bấm dự đoán
    if st.button('Dự đoán ngay'):
        with st.spinner('Đang phân tích...'):
            # 4. Tiền xử lý ảnh cho đúng chuẩn MobileNetV2
            # Resize về 224x224
            img = image_data.resize((224, 224))
            # Chuyển thành array
            x = image.img_to_array(img)
            # Thêm chiều batch (1, 224, 224, 3)
            x = np.expand_dims(x, axis=0)
            # Preprocess (chuẩn hóa pixel)
            x = preprocess_input(x)

            # 5. Dự đoán
            preds = model.predict(x)
            # Lấy top 3 kết quả
            decoded_preds = decode_predictions(preds, top=3)[0]
            
            # 6. Logic kiểm tra Chó hay Mèo (Dựa trên nhãn ImageNet)
            # Chúng ta sẽ kiểm tra xem label có chứa từ khóa không
            is_dog = False
            is_cat = False
            top_label = decoded_preds[0][1] # Lấy tên class có xác suất cao nhất
            prob = decoded_preds[0][2]      # Lấy xác suất

            # Danh sách từ khóa
            # Lưu ý: ImageNet chia rất kỹ (VD: 'tabby', 'tiger_cat'...) nên ta check string
           # Danh sách từ khóa Mèo (giữ nguyên hoặc bổ sung thêm)
            cat_keywords = ['cat', 'tabby', 'tiger', 'siamese', 'persian', 'lynx', 'leopard', 'kitten', 'cougar', 'lion', 'panther', 'cheetah', 'jaguar']

            # Danh sách từ khóa Chó (Cập nhật đầy đủ hơn)
            dog_keywords = [
                'dog', 'terrier', 'retriever', 'spaniel', 'shepherd', 'hound', 'boxer', 'bulldog', 'dalmatian', 
                'husky', 'corgi', 'pug', 'pomeranian', 'chihuahua', 'beagle', 'collie', 'poodle', 'rottweiler', 
                'doberman', 'shiba', 'akita', 'malamute', 'samoyed', 'chow', 'dane', 'mastiff', 'bernese', 
                'newfoundland', 'schnauzer', 'pinscher', 'sheepdog', 'pointer', 'vizsla', 'setter', 'maltese', 
                'papillon', 'pekingese', 'spitz', 'whippet', 'basenji', 'borzoi', 'greyhound', 'bloodhound', 'wolf'
            ]

            # Kiểm tra label cao nhất
            check_str = top_label.lower()
            
            # Logic check đơn giản
            if any(k in check_str for k in cat_keywords):
                is_cat = True
            elif any(k in check_str for k in dog_keywords):
                is_dog = True
            
            # Hiển thị kết quả
            st.divider()
            if is_dog:
                st.success(f"Kết quả: ĐÂY LÀ CHÓ (DOG)  - ({top_label})")
                st.progress(float(prob))
            elif is_cat:
                st.success(f"Kết quả: ĐÂY LÀ MÈO (CAT)  - ({top_label})")
                st.progress(float(prob))
            else:
                # Nếu không phải chó/mèo (trường hợp user up ảnh xe cộ, người...)
                st.warning(f"Hmm... Hình như không phải chó hay mèo. Máy dự đoán là: {top_label}")
