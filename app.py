import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Foldscope Image Processing Dashboard", layout="wide")

st.title("🔬 Foldscope Image Enhancement & Edge Detection Dashboard")

st.write("Upload a microscope image to enhance it and detect microorganisms or particles.")

# Upload image
uploaded_file = st.file_uploader("Upload Microscope Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Unable to read image. Please upload a valid file.")
    else:

        # Convert to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Noise removal
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blur)

        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)

        # Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_img = img_rgb.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:   # filter small noise
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(contour_img,(x,y),(x+w,y+h),(0,255,0),2)

        # Display images
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(img_rgb)

            st.subheader("Enhanced Image")
            st.image(enhanced, channels="GRAY")

        with col2:
            st.subheader("Edge Detection")
            st.image(edges, channels="GRAY")

            st.subheader("Detected Particles / Microorganisms")
            st.image(contour_img)

        st.success(f"Detected Objects: {len(contours)}")