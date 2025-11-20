import streamlit as st
import requests
import base64
import os
import sys
from PIL import Image
import io
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import getUtils


utils = getUtils()
config = utils.load_yaml()
# ================================================================
# 🔹 API CLIENT
# ================================================================
class FishDetectionAPI:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def detect(self, file_bytes: bytes):
        """Send image to FastAPI for fish detection."""
        try:
            response = requests.post(
                self.api_url,
                files={"file": file_bytes}
            )
            if response.status_code == 200:
                return response.json(), None
            return None, f"API Error: {response.status_code}"
        except Exception as e:
            return None, str(e)


# ================================================================
# 🔹 UI COMPONENTS
# ================================================================
class UIComponents:

    @staticmethod
    def apply_custom_css():
        """Apply custom styling to the UI."""
        st.markdown("""
        <style>
        body {
            background-color: #F5F7FA;
        }
        .title {
            font-size: 38px;
            color: #0B3C5D;
            font-weight: 800;
        }
        .subtitle {
            font-size: 18px;
            color: #3A506B;
        }
        .box {
            padding: 12px;
            background: white;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_title():
        """Render page title and subtitle."""
        st.markdown("<h1 class='title'>🐟 Underwater Fish Detection System</h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p class='subtitle'>Upload an image → FastAPI processes → Streamlit displays results</p>",
            unsafe_allow_html=True
        )
        st.write("")

    @staticmethod
    def file_uploader():
        """Render file uploader."""
        return st.file_uploader("Upload Underwater Image", type=["jpg", "jpeg", "png"])

    @staticmethod
    def show_original(col, uploaded):
        with col:
            st.markdown("### 📷 Original Image")
            st.image(uploaded, use_container_width=True)

    @staticmethod
    def show_annotated(col, image):
        with col:
            st.markdown("### 🔍 YOLO Detection")
            st.image(image, use_container_width=True)

    @staticmethod
    def show_summary(data):
        st.markdown("### 📊 Detection Summary")
        st.markdown(f"#### Total Fish: **{data['total_fish']}**")

        # st.write("---")
        for sp, c in data["species_count"].items():
            st.markdown(f"##### - {sp}: {c}")

        # Show GradCAM / max confidence if available
        if "gradcam_score" in data and data["gradcam_score"] is not None:
            st.markdown(f"##### GradCAM Max Confidence: **{data['gradcam_score']:.2f}**")

        




# ================================================================
# 🔹 MAIN STREAMLIT APP
# ================================================================
class FishDetectionApp:

    def __init__(self, api_url: str):
        self.api = FishDetectionAPI(api_url)
        UIComponents.apply_custom_css()
        UIComponents.render_title()

    @staticmethod
    def decode_image(base64_str: str):
        """Decode base64 image returned from backend."""
        annotated_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(annotated_bytes))

    def run(self):
        """Run the Streamlit UI app."""
        uploaded = UIComponents.file_uploader()

        if not uploaded:
            return

        # UI layout → 3 columns: original, detection, gradcam
        col1, col2, col3 = st.columns([1, 1, 1])

        # --------------------------------------------
        # 1️⃣ ORIGINAL IMAGE
        # --------------------------------------------
        UIComponents.show_original(col1, uploaded)

        # --------------------------------------------
        # CALL FASTAPI BACKEND
        # --------------------------------------------
        with st.spinner("⏳ Detecting fish..."):
            data, err = self.api.detect(uploaded.getvalue())

        if err:
            st.error(f"❌ Detection failed: {err}")
            return

        # --------------------------------------------
        # 2️⃣ YOLO DETECTION
        # --------------------------------------------
        detected_img = self.decode_image(data["annotated_image"])
        UIComponents.show_annotated(col2, detected_img)

        # --------------------------------------------
        # 3️⃣ EXPLAINABLE AI (GradCAM)
        # --------------------------------------------
        heatmap_img = self.decode_image(data["heatmap_image"])
        with col3:
            st.markdown("### 🔥 Explainable AI")
            st.image(heatmap_img, use_container_width=True)

        # --------------------------------------------
        # SUMMARY
        # --------------------------------------------
        UIComponents.show_summary(data)

        st.success("🎉 Detection Completed Successfully!")


# ================================================================
# 🔹 RUN THE APP
# ================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="🐟 Fish Detection System", layout="wide")
    app = FishDetectionApp(api_url= config['api_url']['API_URL'])
    app.run()