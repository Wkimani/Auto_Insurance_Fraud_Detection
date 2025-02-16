# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:23:24 2024

@author: WKimani
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import urllib.request
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve


# Define the function to load the model and encoder dictionary
@st.cache_resource
def load_model():
    # Define file names
    model_file = "optimized_model.pkl"
    scaler_file = "scaler.pkl"  # Separate scaler file
    encoder_file = "encoder.sav"

    # Load the model
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)

    # Load the scaler
    with open(scaler_file, 'rb') as file:
        scaler = pickle.load(file)

    # Load the encoder dictionary
    with open(encoder_file, 'rb') as file:
        encoder_dict = pickle.load(file)  # Load as a dictionary of encoders

    return loaded_model, scaler, encoder_dict

# Main app function
def main():
    st.title("Auto Insurance Fraud Prediction")
    st.write("""
    This application will predict whether an auto insuarance claim is fraudulent or genuine based on the provided information. Please fill in the details below and click predict to see the result.
    """)
    
    # Load the trained model and encoder dictionary
    loaded_model, scaler, encoder_dict = load_model()
    
    # Define selection options for user inputs
    months_of_year = (
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    )
    days_of_week = (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    )
    AccidentArea = ("Urban", "Rural")
    Sex = ("Male", "Female")
    Marital_status = ('Single', 'Married', 'Widow', 'Divorced')
    Policy_type = ('Policy Holder', 'Third Party')
    VehicleCategory = ('Sport', 'Sedan', 'Utility')
    Days_Policy_Accident = ('more than 30', '15 to 30', 'none', '1 to 7', '8 to 15')
    Days_Policy_Claim = ('more than 30', '15 to 30', '8 to 15')
    PoliceReportFiled = ('No', 'Yes')
    WitnessPresent = ('No', 'Yes')
    AgentType = ('External', 'Internal')
    AddressChange_Claim = ('1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months')
    NumberOfCars = ('1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8')
    VehiclePrice = ('less than 20k', '20k to 30k', '30k to 40k', '40k to 50k', 'more than 50k')
    AgeOfVehicle = ('new', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years or more')
    Deductible = ('500', '1000', '1500', '2000')
    DriverRating = ('1', '2', '3', '4', '5')
    PastNumberOfClaims = ('0', '1', '2', '3', '4 or more')
    AgeOfPolicyHolder = ('18-25', '26-35', '36-45', '46-55', '56-65', '66 or older')
    NumberOfSuppliments = ('0', '1', '2', '3', '4 or more')
    Make = (
        'Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac',
        'Acura', 'Dodge', 'Mercury', 'Jaguar', 'Nissan', 'VW', 
        'Saab', 'Saturn', 'Porsche', 'BMW', 'Mercedes', 'Ferrari', 'Lexus'
    )
    BasePolicy = ('Liability', 'Collision', 'All Perils')
    Fault = ('Third Party', 'Policy Holder')
    Year = ('1994', '1995', '1996')
    # Collect user inputs
    st.subheader("Policy Information")
    RepNumber = st.text_input("Customer Policy Number")
    Policy_type_selected = st.selectbox('Policy Type', Policy_type)
    Days_Policy_Accident_selected = st.selectbox('Days Since Policy Accident', Days_Policy_Accident)
    Days_Policy_Claim_selected = st.selectbox('Days Since Policy Claim', Days_Policy_Claim)
    AddressChange_Claim_selected = st.selectbox('Address Change Since Claim', AddressChange_Claim)
    AgentType_selected = st.selectbox('Agent Type', AgentType)
    BasePolicy_selected = st.selectbox('Base of Policy', BasePolicy)
    Fault_selected = st.selectbox('Type of Faulty', Fault)
    
    st.subheader("Vehicle Information")
    Make_selected = st.selectbox('Vehicle Make', Make)
    VehicleCategory_selected = st.selectbox('Vehicle Category', VehicleCategory)
    VehiclePrice_selected = st.selectbox('Vehicle Price', VehiclePrice)
    AgeOfVehicle_selected = st.selectbox('Age of Vehicle', AgeOfVehicle)
    NumberOfCars_selected = st.selectbox('Number of Vehicles Owned', NumberOfCars)
    Deductible_selected = st.selectbox('Deductible Amount', Deductible)
    DriverRating_selected = st.selectbox('Driver Rating', DriverRating)
    
    st.subheader("Accident Information")
    AccidentArea_selected = st.selectbox('Accident Area', AccidentArea)
    PoliceReportFiled_selected = st.selectbox('Police Report Filed', PoliceReportFiled)
    WitnessPresent_selected = st.selectbox('Witness Present', WitnessPresent)
    Month_selected = st.selectbox('Month of Accident', months_of_year)
    #Weekday_selected = st.selectbox('Day of Accident', days_of_week)
    Year_selected = st.selectbox('Year of Accident', Year)
    
    st.subheader("Customer Information")
    Sex_selected = st.selectbox('Gender', Sex)
    Marital_status_selected = st.selectbox('Marital Status', Marital_status)
    AgeOfPolicyHolder_selected = st.selectbox('Age of Policy Holder', AgeOfPolicyHolder)
    PastNumberOfClaims_selected = st.selectbox('Past Number of Claims', PastNumberOfClaims)
    NumberOfSuppliments_selected = st.selectbox('Number of Supplements', NumberOfSuppliments)
    
    # Prepare input data for prediction
    input_data = {
        'AccidentArea': AccidentArea_selected,
        'Sex': Sex_selected,
        'MaritalStatus': Marital_status_selected,
        'Fault': Fault_selected,
        'PolicyType': Policy_type_selected,
        'VehicleCategory': VehicleCategory_selected,
        'Make': Make_selected,
        'DriverRating': DriverRating_selected,
        'Days_Policy_Accident': Days_Policy_Accident_selected,
        'Days_Policy_Claim': Days_Policy_Claim_selected,
        'PoliceReportFiled': PoliceReportFiled_selected,
        'WitnessPresent': WitnessPresent_selected,
        'VehiclePrice': VehiclePrice_selected,
        'AgentType': AgentType_selected,
        'AddressChange_Claim': AddressChange_Claim_selected,
        'PastNumberOfClaims': PastNumberOfClaims_selected,
        'NumberOfSuppliments': NumberOfSuppliments_selected,
        'NumberOfCars': NumberOfCars_selected,
        'Year': Year_selected,
        'BasePolicy': BasePolicy_selected,
        'Month': Month_selected,
        'AgeOfVehicle': AgeOfVehicle_selected,
        'AgeOfPolicyHolder': AgeOfPolicyHolder_selected,
        'RepNumber': RepNumber,
        'Deductible': Deductible_selected,
        #'Weekday': Weekday_selected,
        
}
    
    input_df = pd.DataFrame([input_data])


    # Encoding categorical variables
    try:
        input_df_encoded = input_df.copy()

        for column, enc in encoder_dict.items():
            if column in input_df.columns:
                try:
                    # Ensure the column is a pandas Series
                    if isinstance(input_df[column].iloc[0], list):
                        input_df[column] = pd.Series(input_df[column].iloc[0])
                    
                    # Transform the column using the encoder
                    input_df_encoded[column] = enc.transform(input_df[column])

                except ValueError:  # If unseen labels are found
                    # Convert column to pandas Series if it's a list
                    if isinstance(input_df[column].iloc[0], list):
                        input_df[column] = pd.Series(input_df[column].iloc[0])

                    # Re-fit the encoder on the new data (including unseen labels)
                    enc.fit(input_df[column])
                    input_df_encoded[column] = enc.transform(input_df[column])
        
        print("Encoding successful!")
        print(input_df_encoded)

    except Exception as e:
        print(f"Error in encoding input data: {e}")




    # Retrieve the optimized recall threshold (use default if missing)
    RECALL_THRESHOLD = getattr(loaded_model, 'optimal_threshold', 0.55)  # Default to 0.55s if not found

    # Make prediction
    if st.button("Predict"):
        try:
            # Apply scaling before prediction
            input_df_scaled = scaler.transform(input_df_encoded)

            # Get probability of default
            prediction_proba = loaded_model.predict_proba(input_df_scaled)[:, 1]

            # Apply recall-optimized threshold
            prediction = (prediction_proba >= RECALL_THRESHOLD).astype(int)  # Ensure it's an integer output

            # Adjust Likelihood Calculation
            likelihood = prediction_proba[0] * 100  # Always use fraud probability

            if prediction[0] == 1:  # Predicted as 'Fraudulent Claim'
                result = 'Fraudulent Claim'
                st.success(f"**Prediction: {result}**")
                st.info(f"🔴 **Fraud Risk: {likelihood:.2f}%**")
                st.warning("⚠️ The model suggests this claim has a high risk of being fraudulent. Further investigation is recommended before processing.")

            else:  # Predicted as 'Genuine Claim'
                result = 'Genuine Claim'
                fraud_safety = (100 - likelihood)  # Reverse fraud risk to show safety %
                st.success(f"**Prediction: {result}**")
                st.info(f"🟢 **Confidence in Claim Being Genuine: {fraud_safety:.2f}%**")
                st.success("✅ This claim appears genuine based on the provided details. However, always cross-check with policy records.")


            # Ensure likelihood is within valid range
            likelihood = max(0, min(likelihood, 100))

            # Debugging: Print raw probability values
            #st.write(f"Raw Probability: {prediction_proba[0]:.4f}")
            #st.write(f"Threshold Used: {RECALL_THRESHOLD:.3f}")

            # Display results
            #st.success(f"Prediction: **{result}**")
            #st.info(f"Likelihood: **{likelihood:.2f}%**")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    st.write("---")
    st.write("Developed by Wangari Kimani")

if __name__ == '__main__':
    main()
