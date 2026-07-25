import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# ==========================
# Load Model and Files
# ==========================
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("one_hot_encoder_geography.pkl", "rb") as f:
    one_hot_encoder_geography = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================
# Streamlit UI
# ==========================
st.title("Customer Churn Prediction")

geography = st.selectbox(
    "Geography",
    one_hot_encoder_geography.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# ==========================
# Prediction Button
# ==========================
if st.button("Predict"):

    # Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Input DataFrame
    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-Hot Encode Geography
    geo_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder_geography.get_feature_names_out(["Geography"])
    )

    # Combine Data
    final_input = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

    # Reorder Columns (important)
    if hasattr(scaler, "feature_names_in_"):
        final_input = final_input.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale Data
    final_input_scaled = scaler.transform(final_input)

    # Prediction
    probability = model.predict(final_input_scaled)[0][0]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {probability:.2%}")

    if probability > 0.5:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")