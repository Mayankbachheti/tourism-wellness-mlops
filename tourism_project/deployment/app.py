import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
import joblib

MODEL_REPO = "BigBachheti/tourism-wellness-model"  # <-- update with your HF username


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="best_model_v1.joblib", repo_type="model"
    )
    return joblib.load(model_path)


model = load_model()

st.title("Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a customer's details below to predict whether they are likely to "
    "purchase the newly-launched Wellness Tourism Package, before the sales "
    "team spends time on a follow-up call."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 130, 15)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    number_of_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    number_of_trips = st.number_input("Number of Trips per Year", 0, 25, 3)
    passport = st.selectbox("Holds Passport", [0, 1])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car", [0, 1])
    number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000, 100000, 22000)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Package — probability: {probability:.1%}")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package — probability: {probability:.1%}")
