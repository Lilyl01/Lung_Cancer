import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Load the trained models
# lr_model = load('logreg_model.joblib')
rf_model = load('rf_model.joblib')
# svm_model = load('svm_model.joblib')
scaler = load('scaler.pkl') 
# scaler = MinMaxScaler()

# Mappings for numerical encoding to 'YES'/'NO'
mapping = {
    1: 'YES',
    2: 'NO'
}

# ---------------- Dataset Preview Page ----------------
def dataset_preview_page():
    st.title('📊 DATASET PREVIEW')
    st.header('LUNG CANCER PREDICTION DATASET')

    # Link to dataset
    dataset_link = 'https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer'
    st.write(f'You can download the full dataset from [Kaggle]({dataset_link}).')

    # Load a sample dataset for preview
    df = pd.read_csv('lung_data.csv')  # Update this with your dataset file
    st.write('HERE IS A PREVIEW OF THE DATASET:')
    st.dataframe(df.head(20))

# ---------------- Prediction Page ----------------
def prediction_page():
    st.title('🫁 LUNG CANCER PREDICTION APP')
    st.write('FILL IN THE PATIENT DETAILS TO PREDICT THE RISK OF LUNG CANCER.')

    # Input fields for user data
    AGE = st.number_input('AGE 🎂', min_value=0, max_value=120, value=50)
    GENDER = st.selectbox('GENDER 👤', ['M', 'F'])
    SMOKING = st.selectbox('DO YOU SMOKE? 🚬', ['YES', 'NO'])
    YELLOW_FINGERS = st.selectbox('YELLOW FINGERS ✋', ['YES', 'NO'])
    ANXIETY = st.selectbox('ANXIETY 😟', ['YES', 'NO'])
    PEER_PRESSURE = st.selectbox('PEER PRESSURE 👥', ['YES', 'NO'])
    CHRONIC_DISEASE = st.selectbox('CHRONIC DISEASE 🏥', ['YES', 'NO'])
    FATIGUE = st.selectbox('FATIGUE 😴', ['YES', 'NO'])
    ALLERGY = st.selectbox('ALLERGY 🤧', ['YES', 'NO'])
    WHEEZING = st.selectbox('WHEEZING 😤', ['YES', 'NO'])
    ALCOHOL_CONSUMPTION = st.selectbox('ALCOHOL CONSUMPTION 🍺', ['YES', 'NO'])
    COUGHING = st.selectbox('COUGHING 🤧', ['YES', 'NO'])
    SHORTNESS_OF_BREATH = st.selectbox('SHORTNESS OF BREATH 🫁', ['YES', 'NO'])
    SWALLOWING_DIFFICULTY = st.selectbox('SWALLOWING DIFFICULTY 😣', ['YES', 'NO'])
    CHEST_PAIN = st.selectbox('CHEST PAIN ❤️‍🩹', ['YES', 'NO'])

    # When user clicks Predict button
    if st.button('PREDICT 🔮'):
        # Create a dictionary for the input
        input_data = {
            'AGE': [AGE],
            'GENDER': [GENDER],
            'SMOKING': [SMOKING],
            'YELLOW_FINGERS': [YELLOW_FINGERS],
            'ANXIETY': [ANXIETY],
            'PEER_PRESSURE': [PEER_PRESSURE],
            'CHRONIC_DISEASE': [CHRONIC_DISEASE],
            'FATIGUE': [FATIGUE],
            'ALLERGY': [ALLERGY],
            'WHEEZING': [WHEEZING],
            'ALCOHOL_CONSUMPTION': [ALCOHOL_CONSUMPTION],
            'COUGHING': [COUGHING],
            'SHORTNESS_OF_BREATH': [SHORTNESS_OF_BREATH],
            'SWALLOWING_DIFFICULTY': [SWALLOWING_DIFFICULTY],
            'CHEST_PAIN': [CHEST_PAIN]
        }

        input_df = pd.DataFrame(input_data)

        # Convert numerical features to 'YES'/'NO'
        input_df['SMOKING'] = input_df['SMOKING'].map({'YES': 1, 'NO': 2})
        input_df['YELLOW_FINGERS'] = input_df['YELLOW_FINGERS'].map({'YES': 1, 'NO': 2})
        input_df['ANXIETY'] = input_df['ANXIETY'].map({'YES': 1, 'NO': 2})
        input_df['PEER_PRESSURE'] = input_df['PEER_PRESSURE'].map({'YES': 1, 'NO': 2})
        input_df['CHRONIC_DISEASE'] = input_df['CHRONIC_DISEASE'].map({'YES': 1, 'NO': 2})
        input_df['FATIGUE'] = input_df['FATIGUE'].map({'YES': 1, 'NO': 2})
        input_df['ALLERGY'] = input_df['ALLERGY'].map({'YES': 1, 'NO': 2})
        input_df['WHEEZING'] = input_df['WHEEZING'].map({'YES': 1, 'NO': 2})
        input_df['ALCOHOL_CONSUMPTION'] = input_df['ALCOHOL_CONSUMPTION'].map({'YES': 1, 'NO': 2})
        input_df['COUGHING'] = input_df['COUGHING'].map({'YES': 1, 'NO': 2})
        input_df['SHORTNESS_OF_BREATH'] = input_df['SHORTNESS_OF_BREATH'].map({'YES': 1, 'NO': 2})
        input_df['SWALLOWING_DIFFICULTY'] = input_df['SWALLOWING_DIFFICULTY'].map({'YES': 1, 'NO': 2})
        input_df['CHEST_PAIN'] = input_df['CHEST_PAIN'].map({'YES': 1, 'NO': 2})

        # Encode categorical variables into model-friendly format
        input_df['GENDER_F'] = (input_df['GENDER'] == 'F').astype(int)
        input_df['GENDER_M'] = (input_df['GENDER'] == 'M').astype(int)

        # Drop GENDER column
        input_df = input_df.drop('GENDER', axis=1)

        # Ensure that all features match the model input
        input_df_scaled = scaler.transform(input_df)

        # Predict using the trained model
        prediction = rf_model.predict(input_df_scaled)[0]

        # Display prediction result
        st.success(f'🌟 PREDICTION: {"HIGH RISK OF LUNG CANCER" if prediction == 1 else "LOW RISK OF LUNG CANCER"}')

# ---------------- About Page ----------------
def about_page():
    st.title('📚 ABOUT THE PROJECT')
    st.header('LUNG CANCER PREDICTION USING MACHINE LEARNING MODELS')
    st.write("""
    THIS PROJECT AIMS TO PREDICT THE LIKELIHOOD OF LUNG CANCER BASED ON PATIENT HEALTH DATA 
    USING A RANDOM FOREST MODEL. THE DATASET INCLUDES RISK FACTORS SUCH AS SMOKING HABITS, 
    MEDICAL HISTORY, AND RESPIRATORY SYMPTOMS.

    THE GOAL IS TO ASSIST HEALTHCARE PROFESSIONALS IN IDENTIFYING INDIVIDUALS 
    AT HIGH RISK EARLY, SUPPORTING PREVENTIVE CARE AND EARLY DIAGNOSIS.
    """)

# ---------------- Main Function ----------------
def main():
    st.sidebar.title('🗂️ NAVIGATION')
    menu_options = ['PREDICTION PAGE', 'DATASET PREVIEW', 'ABOUT THE PROJECT']
    choice = st.sidebar.selectbox('GO TO', menu_options)

    if choice == 'PREDICTION PAGE':
        prediction_page()
    elif choice == 'DATASET PREVIEW':
        dataset_preview_page()
    elif choice == 'ABOUT THE PROJECT':
        about_page()

if __name__ == '__main__':
    main()








