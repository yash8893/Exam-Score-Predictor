import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="🎓",
    layout="centered"
)


# =========================================================
# LOAD MODEL + LABEL ENCODERS
# =========================================================

@st.cache_resource
def load_model_and_encoders():

    # Load the native XGBoost model
    model = XGBRegressor()
    model.load_model("best_xgbooster.json")

    # Load saved LabelEncoders
    label_encoders = joblib.load("label_encoders.pkl")

    return model, label_encoders


try:

    model, label_encoders = load_model_and_encoders()

except Exception as e:

    st.error("❌ Unable to load the model or label encoders.")

    st.code(str(e))

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🎓 Exam Score Predictor")

st.write(
    "Predict an estimated exam score using a tuned "
    "XGBoost Regression model."
)

st.divider()


# =========================================================
# NUMERICAL INPUTS
# =========================================================

st.subheader("📊 Student Information")

col1, col2 = st.columns(2)

with col1:

    study_hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    class_attendance = st.number_input(
        "Class Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )


with col2:

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )


# =========================================================
# CATEGORICAL INPUTS
# =========================================================

st.subheader("⚙️ Student Preferences")

sleep_quality = st.selectbox(
    "Sleep Quality",
    options=label_encoders["sleep_quality"].classes_
)

study_method = st.selectbox(
    "Study Method",
    options=label_encoders["study_method"].classes_
)

facility_rating = st.selectbox(
    "Facility Rating",
    options=label_encoders["facility_rating"].classes_
)


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Exam Score",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # LABEL ENCODING
    # -----------------------------------------------------

    sleep_quality_encoded = (
        label_encoders["sleep_quality"]
        .transform([sleep_quality])[0]
    )

    study_method_encoded = (
        label_encoders["study_method"]
        .transform([study_method])[0]
    )

    facility_rating_encoded = (
        label_encoders["facility_rating"]
        .transform([facility_rating])[0]
    )


    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    #
    # These are the EXACT 6 features used for training.
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "study_hours": [study_hours],

        "class_attendance": [class_attendance],

        "sleep_hours": [sleep_hours],

        "sleep_quality": [sleep_quality_encoded],

        "study_method": [study_method_encoded],

        "facility_rating": [facility_rating_encoded]

    })


    # -----------------------------------------------------
    # ENSURE EXACT FEATURE ORDER
    # -----------------------------------------------------

    expected_features = [
        "study_hours",
        "class_attendance",
        "sleep_hours",
        "sleep_quality",
        "study_method",
        "facility_rating"
    ]

    input_data = input_data[expected_features]


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]


    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    st.success(
        f"## 🎯 Predicted Exam Score: {prediction:.2f}"
    )


    # -----------------------------------------------------
    # PERFORMANCE INTERPRETATION
    # -----------------------------------------------------

    if prediction >= 90:

        st.info("🌟 Excellent predicted performance!")

    elif prediction >= 75:

        st.info("👍 Good predicted performance.")

    elif prediction >= 50:

        st.warning(
            "📚 Moderate predicted performance. "
            "More preparation may help."
        )

    else:

        st.error(
            "⚠️ Low predicted score. "
            "Consider improving your study habits."
        )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📋 Model Information")

st.sidebar.write(
    "**Model:** XGBoost Regressor"
)

st.sidebar.write(
    "**Target:** Exam Score"
)

st.sidebar.write(
    "**Preprocessing:** Label Encoding"
)

st.sidebar.write(
    "**Features:** 6"
)

st.sidebar.write(
    "**Scaling:** Not Required"
)