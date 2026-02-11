import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# PROJECT IMPORTS
from utils.model_utils import load_model, predict_image, CLASS_NAMES
from db.db_utils import init_db, insert_record, fetch_all_records

# PAGE CONFIG 
st.set_page_config(
    page_title="Urban Tree Health Monitoring",
    page_icon="🌳",
    layout="wide"
)

# INITIALIZE DATABASE 
init_db()

# LOAD MODEL 
MODEL_PATH = "model/mobilenetv2_tree_health.pth"

@st.cache_resource
def load_cached_model():
    return load_model(MODEL_PATH)

model = load_cached_model()

# HEADER 
st.markdown(
    "<h1 style='text-align:center;'>🌳 Urban Tree Health Monitoring System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Offline AI-based Tree Health Assessment using Computer Vision</p>",
    unsafe_allow_html=True
)

st.divider()

# SIDEBAR 
st.sidebar.title(" Project Overview")
st.sidebar.markdown("""
**Model:** MobileNetV2 (Transfer Learning)  
**Health Classes:**  
• Healthy  
• Moderate / Stressed  
• Unhealthy / Diseased  

**System:** Fully Offline  
**Use Case:** Urban Environmental Monitoring  
""")

# MAIN LAYOUT 
col1, col2 = st.columns([1, 1])

# INPUT COLUMN 
with col1:
    st.subheader("📤 Upload Tree Image")

    area_name = st.text_input(
        "📍 Enter Area / Ward Name",
        placeholder="e.g., Zoo Road, Ward 12"
    )

    uploaded_file = st.file_uploader(
        "Upload a tree image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# OUTPUT COLUMN 
with col2:
    st.subheader("Prediction Result")

    if uploaded_file and area_name.strip():
        result = predict_image(model, image)

        st.markdown(
            f"""
            <h3>🌱 Health Status:
            <span style='color:green'>{result['label']}</span></h3>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{result['confidence']*100:.2f}%"
        )

        # PROBABILITY BAR CHART 
        st.markdown("### Class Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, result["all_confidences"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence per Class")

        st.pyplot(fig)

        # SAVE TO DATABASE 
        if st.button("💾 Save Result"):
            insert_record(
                image_name=uploaded_file.name,
                area_name=area_name,
                predicted_health=result["label"],
                confidence=result["confidence"]
            )
            st.success("Prediction saved successfully.")

    elif uploaded_file:
        st.warning("Please enter the area name to save the result.")
    else:
        st.info("Upload an image to get prediction.")

# STORED RECORDS 
st.divider()
st.subheader("🗂️ Stored Tree Health Records")

records = fetch_all_records()

if records:
    df = pd.DataFrame(
        records,
        columns=[
            "ID",
            "Image Name",
            "Area",
            "Predicted Health",
            "Confidence",
            "Timestamp"
        ]
    )

    st.dataframe(df, use_container_width=True)
else:
    st.info("No records found in database yet.")

# AREA-WISE ANALYTICS 
st.divider()
st.subheader(" Area-wise Tree Health Analytics")

if records:
    area_list = df["Area"].unique().tolist()

    selected_area = st.selectbox(
        "📍 Select Area",
        area_list
    )

    area_df = df[df["Area"] == selected_area]

    # BAR CHART 
    st.markdown("### 🌳 Health Status Distribution")

    health_counts = area_df["Predicted Health"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.bar(health_counts.index, health_counts.values)
    ax1.set_ylabel("Number of Trees")
    ax1.set_xlabel("Health Status")
    ax1.set_title(f"Tree Health Distribution in {selected_area}")

    st.pyplot(fig1)

    # PIE CHART 
    st.markdown("### 🥧 Healthy vs At-Risk Trees")

    pie_data = area_df["Predicted Health"].replace(
        {"Moderate / Stressed": "Unhealthy / At Risk"}
    ).value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(
        pie_data.values,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax2.axis("equal")
    ax2.set_title(f"Risk Overview for {selected_area}")

    st.pyplot(fig2)

    # CONFIDENCE TREND 
    st.markdown("### Prediction Confidence Trend")

    area_df_sorted = area_df.sort_values("Timestamp")

    fig3, ax3 = plt.subplots()
    ax3.plot(
        area_df_sorted["Timestamp"],
        area_df_sorted["Confidence"],
        marker="o"
    )
    ax3.set_ylabel("Confidence")
    ax3.set_xlabel("Time")
    ax3.set_title(f"Confidence Trend in {selected_area}")
    ax3.tick_params(axis='x', rotation=45)

    st.pyplot(fig3)

else:
    st.info("No analytics available yet. Please save predictions first.")

# FOOTER 
st.markdown(
    """
    <p style='text-align:center; font-size:12px;'>
    Urban Tree Health Monitoring System <br>
    Offline AI Project – Computer Vision & Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)
