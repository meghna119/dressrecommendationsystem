import streamlit as st
import pandas as pd
import pickle

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2

class BodyClassifierApp:
    def __init__(self):
        # Load the trained model
        self.rf_model = self.load_rf_model()

    def load_rf_model(self):
        try:
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file, encoding='utf-8')
            return model
        except Exception as e:
            st.error(f"Failed to load the RandomForestClassifier model: {e}")
            return None

    def classify(self, gender, age, measurements):
        try:
            # Create DataFrame from user inputs
            data = pd.DataFrame(columns=['Gender', 'Age', 'Shoulder', 'Waist', 'Hips', 'Bust', 'Chest'])
            data.loc[0] = [FEMALE_GENDER if gender == "Female" else MALE_GENDER, age] + measurements

            # Predict body type using RandomForestClassifier
            if self.rf_model:
                body_type = self.rf_model.predict(data)
                return body_type[0]
            else:
                st.error("RandomForestClassifier Model not loaded.")
                return None
        except Exception as e:
            st.error(str(e))
            return None

def main():
    st.title("Body Measurement Classifier")

    # Gender selection
    gender = st.selectbox("Gender:", ["Female", "Male"])

    # Age input
    age = st.number_input("Age:", min_value=0, max_value=150, step=1)

    # Measurement inputs
    measurements = []
    st.header("Measurements")
    for label in ["Shoulder", "Waist", "Hips", "Bust", "Chest"]:
        value = st.number_input(label + ":", step=0.1)
        measurements.append(value)

    # Classify button
    classifier = BodyClassifierApp()
    if st.button("Classify"):
        body_type = classifier.classify(gender, age, measurements)
        if body_type is not None:
            st.success(f"Predicted Body Type: {body_type}")

if __name__ == "__main__":
    main()
