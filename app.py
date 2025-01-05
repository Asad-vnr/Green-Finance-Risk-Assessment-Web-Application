import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('SP 500 ESG Risk Ratings.csv')

# Normalize the dataset to make comparisons easier
scaler = MinMaxScaler()
columns_to_normalize = ['Total ESG Risk score', 'Environment Risk Score', 'Governance Risk Score', 'Social Risk Score']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Streamlit UI: Ask input from the user
st.title("Green Finance Risk Assessment")
company_name = st.text_input("Enter the New Company's Name:")

# User inputs for ESG-related factors (based on user input)
energy_consumption = st.slider("Energy Consumption (1 to 10, where 1 is very low and 10 is very high):", 1, 10)
carbon_footprint = st.slider("Carbon Footprint (1 to 10, where 1 is very low and 10 is very high):", 1, 10)
waste_management = st.slider("Waste Management Efficiency (1 to 10, where 1 is very low and 10 is very high):", 1, 10)

leadership_quality = st.slider("Leadership Quality (1 to 10, where 1 is very poor and 10 is excellent):", 1, 10)
compliance = st.slider("Compliance with Regulations (1 to 10, where 1 is very low and 10 is very high):", 1, 10)
transparency = st.slider("Transparency in Operations (1 to 10, where 1 is very low and 10 is very high):", 1, 10)

employee_satisfaction = st.slider("Employee Satisfaction (1 to 10, where 1 is very low and 10 is very high):", 1, 10)
diversity = st.slider("Diversity in Workforce (1 to 10, where 1 is very low and 10 is very high):", 1, 10)
community_engagement = st.slider("Community Engagement (1 to 10, where 1 is very low and 10 is very high):", 1, 10)

# Function to calculate individual ESG scores based on user input
def calculate_user_esg_scores(energy_consumption, carbon_footprint, waste_management, 
                               leadership_quality, compliance, transparency, 
                               employee_satisfaction, diversity, community_engagement):
    environmental_score = (energy_consumption * 0.3) + (carbon_footprint * 0.4) + (waste_management * 0.3)
    governance_score = (leadership_quality * 0.4) + (compliance * 0.3) + (transparency * 0.3)
    social_score = (employee_satisfaction * 0.4) + (diversity * 0.3) + (community_engagement * 0.3)
    
    total_esg_score = (environmental_score + governance_score + social_score) / 3
    return environmental_score, governance_score, social_score, total_esg_score

# Calculate user's ESG scores
if company_name:
    environmental_score, governance_score, social_score, total_esg_score = calculate_user_esg_scores(
        energy_consumption, carbon_footprint, waste_management, 
        leadership_quality, compliance, transparency, 
        employee_satisfaction, diversity, community_engagement
    )
    
    # Normalize the user's ESG scores using the same scaler
    user_scores = np.array([[environmental_score, governance_score, social_score, total_esg_score]])
    user_scores_scaled = scaler.transform(user_scores)
    
    # Calculate Euclidean distances to each company's scores in the dataset
    df['Euclidean_Distance'] = np.sqrt(
        (df['Environment Risk Score'] - user_scores_scaled[0][0])**2 +
        (df['Governance Risk Score'] - user_scores_scaled[0][1])**2 +
        (df['Social Risk Score'] - user_scores_scaled[0][2])**2 +
        (df['Total ESG Risk score'] - user_scores_scaled[0][3])**2
    )
    
    # Find the closest match (min distance)
    closest_match = df.loc[df['Euclidean_Distance'].idxmin()]

    # Display the result to the user
    st.write(f"Risk Assessment for {company_name}:")
    st.write(f"Environmental Risk Score: {environmental_score:.2f}")
    st.write(f"Governance Risk Score: {governance_score:.2f}")
    st.write(f"Social Risk Score: {social_score:.2f}")
    st.write(f"Total ESG Risk Score: {total_esg_score:.2f}")
    
    st.write(f"\nComparing with the closest match in the dataset: {closest_match['Name']}")
    st.write(f"Closest Company ESG Risk Score: {closest_match['Total ESG Risk score'] * 100:.2f}")
    
    # Investment risk determination
    if total_esg_score <= 4:
        st.write("This company is considered risk-free for investment.")
    elif total_esg_score <= 6:
        st.write("This company has moderate risk for investment.")
    else:
        st.write("This company is high risk for investment.")
