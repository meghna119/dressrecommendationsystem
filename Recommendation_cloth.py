#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2

# Define functions to map body types
def map_body_type_female(row):
    if (row['Bust'] - row['Hips']) <= 1 and (row['Hips'] - row['Bust']) < 3.6 and (row['Bust'] - row['Waist']) >= 9 or (row['Hips'] - row['Waist']) >= 10:
        return 'HOURGLASS'
    elif (row['Bust'] - row['Hips']) >= 3.6 and (row['Bust'] - row['Waist']) < 9:
        return 'INVERTED TRIANGLE'
    elif (row['Hips'] - row['Bust']) < 3.6 and (row['Bust'] - row['Hips']) < 3.6 and (row['Bust'] - row['Waist']) < 9 and (row['Hips'] - row['Waist']) < 10:
        return 'RECTANGLE'
    elif (row['Hips'] - row['Bust']) < 3.6 and (row['Hips'] - row['Waist']) >= 9:
        return 'PEAR'
    elif (row['Hips'] - row['Bust']) >= 3.6 and (row['Hips'] - row['Waist']) < 9:
        return 'APPLE'
    else:
        return 'Other'

def map_body_type_male(row):
    if (row['Waist'] - row['Hips']) >= 0.9 and (row['Waist'] - row['Chest']) >= 0.9 and (row['Waist'] / row['Hips']) >= 0.85 and (row['Waist'] / row['Chest']) >= 0.85:
        return 'RECTANGLE'
    elif (row['Waist'] / row['Hips']) >= 0.9 and (row['Waist'] / row['Hips']) <= 1.0 and (row['Waist'] / row['Chest']) >= 0.9 and (row['Waist'] / row['Chest']) <= 1.0:
        return 'TRIANGLE'
    elif (row['Waist'] / row['Hips']) <= 0.9 and (row['Shoulder'] / row['Hips']) >= 1.2:
        return 'INVERTED TRIANGLE'
    elif (row['Waist'] / row['Chest']) <= 0.9 and (row['Shoulder'] / row['Waist']) >= 1.2:
        return 'OVAL'
    elif (row['Waist'] / row['Hips']) >= 1.1 and (row['Waist'] / row['Chest']) >= 1.1:
        return 'OVAL'
    elif (row['Waist'] / row['Hips']) >= 0.9 and (row['Waist'] / row['Hips']) <= 1.1 and (row['Waist'] / row['Chest']) >= 0.9 and (row['Waist'] / row['Chest']) <= 1.1 and (row['Shoulder'] / row['Waist']) >= 1.1 and (row['Shoulder'] / row['Hips']) >= 1.1:
        return 'TRAPEZOID'
    else:
        return 'Other'

def map_body_type(row):
    gender = row['Gender']
    if gender == FEMALE_GENDER:
        return map_body_type_female(row)
    elif gender == MALE_GENDER:
        return map_body_type_male(row)
    else:
        return 'Other'

# Load dataset
data = pd.read_csv('Body_Measurement.csv')

# Apply mapping function
data['Body_Type'] = data.apply(map_body_type, axis=1)

# Prepare features and target
X = data[['Gender', 'Age', 'Shoulder', 'Waist', 'Hips', 'Bust', 'Chest']]
y = data['Body_Type']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define feature names
feature_names = ['Gender', 'Age', 'Shoulder', 'Waist', 'Hips', 'Bust', 'Chest']

# Define and fit the Random Forest classifier with feature names
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print accuracy and classification report
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[5]:





# In[4]:


import pickle


# In[7]:


with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[14]:


import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2

class BodyClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Body Measurement Classifier")

        # Gender selection
        self.gender_label = tk.Label(master, text="Gender:")
        self.gender_label.grid(row=0, column=0, padx=5, pady=5)
        self.gender_var = tk.StringVar(master)
        self.gender_var.set("Female")  # Default value
        self.gender_option = tk.OptionMenu(master, self.gender_var, "Female", "Male")
        self.gender_option.grid(row=0, column=1, padx=5, pady=5)

        # Age input
        self.age_label = tk.Label(master, text="Age:")
        self.age_label.grid(row=1, column=0, padx=5, pady=5)
        self.age_entry = tk.Entry(master)
        self.age_entry.grid(row=1, column=1, padx=5, pady=5)

        # Measurement inputs
        self.measurement_labels = ["Shoulder", "Waist", "Hips", "Bust", "Chest"]
        self.measurement_entries = []

        for i, label in enumerate(self.measurement_labels):
            tk.Label(master, text=label + ":").grid(row=i + 2, column=0, padx=5, pady=5)
            entry = tk.Entry(master)
            entry.grid(row=i + 2, column=1, padx=5, pady=5)
            self.measurement_entries.append(entry)

        # Classify button
        self.classify_button = tk.Button(master, text="Classify", command=self.classify)
        self.classify_button.grid(row=len(self.measurement_labels) + 3, columnspan=2, padx=5, pady=10)

        # Classification result
        self.result_label = tk.Label(master, text="")
        self.result_label.grid(row=len(self.measurement_labels) + 4, columnspan=2, padx=5, pady=5)

        # Load the trained RandomForestClassifier
        self.model = self.load_model()

    def load_model(self):
    # Load the trained RandomForestClassifier
        try:
            with open('random_forest_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            return None


    def classify(self):
        try:
            # Get user inputs
            gender = self.gender_var.get()
            age = float(self.age_entry.get())
            measurements = [float(entry.get()) for entry in self.measurement_entries]

            # Create DataFrame from user inputs
            data = pd.DataFrame(columns=['Gender', 'Age', 'Shoulder', 'Waist', 'Hips', 'Bust', 'Chest'])
            data.loc[0] = [FEMALE_GENDER if gender == "Female" else MALE_GENDER, age] + measurements

            # Predict body type using RandomForestClassifier
            if self.rf_classifier:
                body_type = self.rf_classifier.predict(data)
                self.result_label.config(text=f"Predicted Body Type: {body_type}")
            else:
                messagebox.showerror("Error", "Model not loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = BodyClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[13]:


import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2

# Define functions to map body types for females
def map_body_type_female(row):
    if (row['Bust'] - row['Hips']) <= 1 and (row['Hips'] - row['Bust']) < 3.6 and (row['Bust'] - row['Waist']) >= 9 or (row['Hips'] - row['Waist']) >= 10:
        return 'Hourglass'
    elif (row['Bust'] - row['Hips']) >= 3.6 and (row['Bust'] - row['Waist']) < 9:
        return 'Inverted Triangle'
    elif (row['Hips'] - row['Bust']) < 3.6 and (row['Bust'] - row['Hips']) < 3.6 and (row['Bust'] - row['Waist']) < 9 and (row['Hips'] - row['Waist']) < 10:
        return 'Rectangle'
    elif (row['Hips'] - row['Bust']) < 3.6 and (row['Hips'] - row['Waist']) >= 9:
        return 'Pear'
    elif (row['Hips'] - row['Bust']) >= 3.6 and (row['Hips'] - row['Waist']) < 9:
        return 'Apple'
    else:
        return 'Other'

# Define functions to map body types for males
def map_body_type_male(row):
    if (row['Waist'] - row['Hips']) >= 0.9 and (row['Waist'] - row['Chest']) >= 0.9 and (row['Waist'] / row['Hips']) >= 0.85 and (row['Waist'] / row['Chest']) >= 0.85:
        return 'Rectangle'
    elif (row['Waist'] / row['Hips']) >= 0.9 and (row['Waist'] / row['Hips']) <= 1.0 and (row['Waist'] / row['Chest']) >= 0.9 and (row['Waist'] / row['Chest']) <= 1.0:
        return 'Triangle'
    elif (row['Waist'] / row['Hips']) <= 0.9 and (row['Shoulder'] / row['Hips']) >= 1.2:
        return 'Inverted Triangle'
    elif (row['Waist'] / row['Chest']) <= 0.9 and (row['Shoulder'] / row['Waist']) >= 1.2:
        return 'Oval'
    elif (row['Waist'] / row['Hips']) >= 1.1 and (row['Waist'] / row['Chest']) >= 1.1:
        return 'Oval'
    else: 
        return 'Trapezoid'
    
        

# Define function to map body types based on gender
def map_body_type(row):
    gender = row['Gender']
    if gender == FEMALE_GENDER:
        return map_body_type_female(row)
    elif gender == MALE_GENDER:
        return map_body_type_male(row)
    else:
        return 'Other'

class BodyClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Body Measurement Classifier")

        # Gender selection
        self.gender_label = tk.Label(master, text="Gender:")
        self.gender_label.grid(row=0, column=0, padx=5, pady=5)
        self.gender_var = tk.StringVar(master)
        self.gender_var.set("Female")  # Default value
        self.gender_option = tk.OptionMenu(master, self.gender_var, "Female", "Male")
        self.gender_option.grid(row=0, column=1, padx=5, pady=5)

        # Age input
        self.age_label = tk.Label(master, text="Age:")
        self.age_label.grid(row=1, column=0, padx=5, pady=5)
        self.age_entry = tk.Entry(master)
        self.age_entry.grid(row=1, column=1, padx=5, pady=5)

        # Measurement inputs
        self.measurement_labels = ["Shoulder", "Waist", "Hips", "Bust", "Chest"]
        self.measurement_entries = []

        for i, label in enumerate(self.measurement_labels):
            tk.Label(master, text=label + ":").grid(row=i + 2, column=0, padx=5, pady=5)
            entry = tk.Entry(master)
            entry.grid(row=i + 2, column=1, padx=5, pady=5)
            self.measurement_entries.append(entry)

        # Classify button
        self.classify_button = tk.Button(master, text="Classify", command=self.classify)
        self.classify_button.grid(row=len(self.measurement_labels) + 3, columnspan=2, padx=5, pady=10)

        # Classification result
        self.result_label = tk.Label(master, text="")
        self.result_label.grid(row=len(self.measurement_labels) + 4, columnspan=2, padx=5, pady=5)

    def classify(self):
        try:
            # Get user inputs
            gender = self.gender_var.get()
            age = float(self.age_entry.get())
            measurements = [float(entry.get()) for entry in self.measurement_entries]

            # Create DataFrame from user inputs
            data = pd.DataFrame(columns=['Gender', 'Age', 'Shoulder', 'Waist', 'Hips', 'Bust', 'Chest'])
            data.loc[0] = [FEMALE_GENDER if gender == "Female" else MALE_GENDER, age] + measurements

            # Apply mapping function to categorize body type
            data['Body_Type'] = data.apply(map_body_type, axis=1)

            # Display result
            body_type = data.loc[0, 'Body_Type']
            self.result_label.config(text=f"Predicted Body Type: {body_type}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = BodyClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2

# Define functions to map body types
def map_body_type_female(row):
    if (row['Bust'] - row['Hips']) <= 1 and (row['Hips'] - row['Bust']) < 3.6 and (row['Bust'] - row['Waist']) >= 9 or (row['Hips'] - row['Waist']) >= 10:
        return 'Hourglass'
    elif (row['Bust'] - row['Hips']) >= 3.6 and (row['Bust'] - row['Waist']) < 9:
        return 'Inverted Triangle'
    elif (row['Hips'] - row['Bust']) < 3.6 and (row['Bust'] - row['Hips']) < 3.6 and (row['Bust'] - row['Waist']) < 9 and (row['Hips'] - row['Waist']) < 10:
        return 'Rectangle'
    elif (row['Hips'] - row['Bust']) < 3.6 and (row['Hips'] - row['Waist']) >= 9:
        return 'Pear'
    elif (row['Hips'] - row['Bust']) >= 3.6 and (row['Hips'] - row['Waist']) < 9:
        return 'Apple'
    else:
        return 'Other'

def map_body_type_male(row):
    if (row['Waist'] - row['Hips']) >= 0.9 and (row['Waist'] - row['Chest']) >= 0.9 and (row['Waist'] / row['Hips']) >= 0.85 and (row['Waist'] / row['Chest']) >= 0.85:
        return 'Men Rectangle'
    elif (row['Waist'] / row['Hips']) >= 0.9 and (row['Waist'] / row['Hips']) <= 1.0 and (row['Waist'] / row['Chest']) >= 0.9 and (row['Waist'] / row['Chest']) <= 1.0:
        return 'Men Triangle'
    elif (row['Waist'] / row['Hips']) <= 0.9 and (row['Shoulder'] / row['Hips']) >= 1.2:
        return 'Men Inverted Triangle'
    elif (row['Waist'] / row['Chest']) <= 0.9 and (row['Shoulder'] / row['Waist']) >= 1.2:
        return 'Men Oval'
    elif (row['Waist'] / row['Hips']) >= 1.1 and (row['Waist'] / row['Chest']) >= 1.1:
        return 'Men Oval'
    elif (row['Waist'] / row['Hips']) >= 0.9 and (row['Waist'] / row['Hips']) <= 1.1 and (row['Waist'] / row['Chest']) >= 0.9 and (row['Waist'] / row['Chest']) <= 1.1 and (row['Shoulder'] / row['Waist']) >= 1.1 and (row['Shoulder'] / row['Hips']) >= 1.1:
        return 'Men Trapezoid'
    else:
        return 'Other'

def map_body_type(row):
    gender = row['Gender']
    if gender == FEMALE_GENDER:
        return map_body_type_female(row)
    elif gender == MALE_GENDER:
        return map_body_type_male(row)
    else:
        return 'Other'

# Load dataset
data = pd.read_csv('Body_Measurement.csv')

# Apply mapping function
data['Body_Type'] = data.apply(map_body_type, axis=1)

# Prepare features and target
X = data[['Gender', 'Age', 'Shoulder', 'Waist', 'Hips', 'Bust', 'Chest']]
y = data['Body_Type']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample minority classes
oversampler = RandomOverSampler()
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Define and tune the Random Forest classifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)
best_rf_classifier = grid_search.best_estimator_

# Predict on test data
y_pred = best_rf_classifier.predict(X_test)

# Print accuracy and classification report
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


