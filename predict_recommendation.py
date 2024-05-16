#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
import pickle
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
        try:
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {e}")
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
            if self.model:
                body_type = self.model.predict(data)
                self.result_label.config(text=f"Predicted Body Type: {body_type[0]}")
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


# In[14]:





# In[3]:


import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2
IMG_SIZE = (224, 224)

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

        # Load the trained models
        self.rf_model = self.load_rf_model()
        self.cnn_model = self.load_cnn_model()

        # Load the dataset
        self.dataset = pd.read_csv('dataset.csv')

    def load_rf_model(self):
        try:
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the RandomForestClassifier model: {e}")
            return None

    def load_cnn_model(self):
        try:
            model = load_model('your_cnn_model.h5')
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the CNN model: {e}")
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
            if self.rf_model:
                body_type = self.rf_model.predict(data)
                self.result_label.config(text=f"Predicted Body Type: {body_type[0]}")
                
                # Provide recommendations based on predicted body type
                self.provide_recommendations(body_type[0], gender)
            else:
                messagebox.showerror("Error", "RandomForestClassifier Model not loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def provide_recommendations(self, body_type, gender):
        try:
            # Filter dataset based on predicted body type and gender
            filtered_data = self.dataset[(self.dataset['Gender'] == gender) & (self.dataset[body_type] == 1)]
            
            # Retrieve top 5 images for each cloth pattern
            recommendations = {}
            for cloth_pattern in filtered_data['Cloth Pattern '].unique():
                pattern_data = filtered_data[filtered_data['Cloth Pattern '] == cloth_pattern]
                top_images = pattern_data.head(5)['Image Path'].tolist()
                recommendations[cloth_pattern] = top_images
            
            # Display recommendations
            self.display_recommendations(recommendations)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_recommendations(self, recommendations):
        # Clear any previous recommendations
        if hasattr(self, 'recommendation_frames'):
            for frame in self.recommendation_frames:
                frame.destroy()
        
        # Create recommendation frames
        self.recommendation_frames = []
        row = len(self.measurement_labels) + 5
        col = 0
        for cloth_pattern, images in recommendations.items():
            tk.Label(self.master, text=f"Top 5 images for {cloth_pattern}:").grid(row=row, column=col, padx=5, pady=5)
            row += 1
            frame = tk.Frame(self.master)
            frame.grid(row=row, column=col, padx=5, pady=5)
            self.recommendation_frames.append(frame)
            for i, image_path in enumerate(images):
                img = Image.open(image_path)
                img = img.resize((100, 100), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                img_label = tk.Label(frame, image=img)
                img_label.image = img
                img_label.grid(row=i, column=0, padx=5, pady=5)
            row = len(self.measurement_labels) + 5
            col += 1

def main():
    root = tk.Tk()
    app = BodyClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[4]:


import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2
IMG_SIZE = (224, 224)

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

        # Load the trained models
        self.rf_model = self.load_rf_model()
        self.cnn_model = self.load_cnn_model()

        # Load the dataset
        self.dataset = pd.read_csv('dataset.csv')

    def load_rf_model(self):
        try:
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the RandomForestClassifier model: {e}")
            return None

    def load_cnn_model(self):
        try:
            model = load_model('your_cnn_model.h5')
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the CNN model: {e}")
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
            if self.rf_model:
                body_type = self.rf_model.predict(data)
                self.result_label.config(text=f"Predicted Body Type: {body_type[0]}")
                
                # Provide recommendations based on predicted body type
                self.provide_recommendations(body_type[0], gender)
            else:
                messagebox.showerror("Error", "RandomForestClassifier Model not loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def provide_recommendations(self, body_type, gender):
        try:
            # Filter dataset based on predicted body type and gender
            filtered_data = self.dataset[(self.dataset['Gender'] == gender) & (self.dataset[body_type] == 1)]
            
            # Retrieve top 5 images for each cloth pattern
            recommendations = {}
            for cloth_pattern in filtered_data['Cloth Pattern '].unique():
                pattern_data = filtered_data[filtered_data['Cloth Pattern '] == cloth_pattern]
                top_images = pattern_data.head(5)['Image Path'].tolist()
                recommendations[cloth_pattern] = top_images
            
            # Display recommendations
            self.display_recommendations(recommendations)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_recommendations(self, recommendations):
        # Clear any previous recommendations
        if hasattr(self, 'recommendation_frames'):
            for frame in self.recommendation_frames:
                frame.destroy()

        # Create recommendation frames
        self.recommendation_frames = []
        row = len(self.measurement_labels) + 5
        col = 0
        for cloth_pattern, images in recommendations.items():
            tk.Label(self.master, text=f"Top 5 images for {cloth_pattern}:").grid(row=row, column=col, padx=5, pady=5)
            row += 1
            frame = tk.Frame(self.master)
            frame.grid(row=row, column=col, padx=5, pady=5)
            self.recommendation_frames.append(frame)
            for i, image_path in enumerate(images):
                # Use absolute path and os.path.join()
                full_image_path = os.path.join(os.getcwd(), image_path)
                img = Image.open(full_image_path)
                img = img.resize((100, 100), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                img_label = tk.Label(frame, image=img)
                img_label.image = img
                img_label.grid(row=i, column=0, padx=5, pady=5)
            row = len(self.measurement_labels) + 5
            col += 1

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
import numpy as np
import os
import pickle
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

        # Load the trained model
        self.rf_model = self.load_rf_model()

        # Load the dataset
        self.dataset = pd.read_csv('dataset.csv')

    def load_rf_model(self):
        try:
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the RandomForestClassifier model: {e}")
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
            if self.rf_model:
                body_type = self.rf_model.predict(data)
                self.result_label.config(text=f"Predicted Body Type: {body_type[0]}")
                
                # Provide recommendations based on predicted body type
                self.provide_recommendations(body_type[0], gender)
            else:
                messagebox.showerror("Error", "RandomForestClassifier Model not loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def provide_recommendations(self, body_type, gender):
        try:
            # Filter dataset based on predicted body type and gender
            filtered_data = self.dataset[(self.dataset['Gender'] == gender) & (self.dataset[body_type] == 1)]
            
            # Retrieve top 5 images for each cloth pattern
            recommendations = {}
            for cloth_pattern in filtered_data['Cloth Pattern '].unique():
                pattern_data = filtered_data[filtered_data['Cloth Pattern '] == cloth_pattern]
                top_images = pattern_data.head(5)['Image Path'].tolist()
                recommendations[cloth_pattern] = top_images
            
            # Display recommendations
            self.display_recommendations(recommendations)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_recommendations(self, recommendations):
        # Clear any previous recommendations
        if hasattr(self, 'recommendation_frames'):
            for frame in self.recommendation_frames:
                frame.destroy()

        # Create recommendation frames
        self.recommendation_frames = []
        row = len(self.measurement_labels) + 5
        col = 0
        for cloth_pattern, images in recommendations.items():
            tk.Label(self.master, text=f"Top 5 images for {cloth_pattern}:").grid(row=row, column=col, padx=5, pady=5)
            row += 1
            frame = tk.Frame(self.master)
            frame.grid(row=row, column=col, padx=5, pady=5)
            self.recommendation_frames.append(frame)
            for i, image_path in enumerate(images):
                # Use absolute path and os.path.join()
                full_image_path = os.path.join(os.getcwd(), image_path)
                img = Image.open(full_image_path)
                img = img.resize((100, 100), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                img_label = tk.Label(frame, image=img)
                img_label.image = img
                img_label.grid(row=i, column=0, padx=5, pady=5)
            row = len(self.measurement_labels) + 5
            col += 1

def main():
    root = tk.Tk()
    app = BodyClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[14]:


import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from PIL import Image, ImageTk

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

        # Load the trained model
        self.rf_model = self.load_rf_model()

        # Load the dataset
        self.dataset = pd.read_csv('dataset.csv')

    def load_rf_model(self):
        try:
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the RandomForestClassifier model: {e}")
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
            if self.rf_model:
                body_type = self.rf_model.predict(data)
                self.result_label.config(text=f"Predicted Body Type: {body_type[0]}")
                
                # Provide recommendations based on predicted body type
                self.provide_recommendations(body_type[0], gender)
            else:
                messagebox.showerror("Error", "RandomForestClassifier Model not loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def provide_recommendations(self, body_type, gender):
        try:
            # Filter dataset based on predicted body type and gender
            filtered_data = self.dataset[(self.dataset['Gender'] == gender) & (self.dataset[body_type] == 1)]
            
            # Retrieve top 5 images for each cloth pattern
            recommendations = {}
            for cloth_pattern in filtered_data['Cloth Pattern '].unique():
                pattern_data = filtered_data[filtered_data['Cloth Pattern '] == cloth_pattern]
                top_images = pattern_data[['Image Path']].head(5).values.tolist()
                recommendations[cloth_pattern] = top_images
            
            # Display recommendations
            self.display_recommendations(recommendations)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_recommendations(self, recommendations):
        # Clear any previous recommendations
        if hasattr(self, 'recommendation_frames'):
            for frame in self.recommendation_frames:
                frame.destroy()

        # Create recommendation frames
        self.recommendation_frames = []
        row = len(self.measurement_labels) + 5
        col = 0
        for cloth_pattern, image_paths in recommendations.items():
            tk.Label(self.master, text=f"Top 5 images for {cloth_pattern}:").grid(row=row, column=col, padx=5, pady=5)
            row += 1
            frame = tk.Frame(self.master)
            frame.grid(row=row, column=col, padx=5, pady=5)
            self.recommendation_frames.append(frame)
            for i, image_path in enumerate(image_paths):
                # Use absolute path and os.path.join()
                full_image_path = os.path.join(os.getcwd(), image_path[0])
                img = Image.open(full_image_path)
                img = img.resize((100, 100), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                img_label = tk.Label(frame, image=img)
                img_label.image = img
                img_label.grid(row=i, column=0, padx=5, pady=5)
            row = len(self.measurement_labels) + 5
            col += 1

def main():
    root = tk.Tk()
    app = BodyClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


# In[ ]:


import streamlit as st
from PIL import Image

# Constants
FEMALE_GENDER = 1
MALE_GENDER = 2

class BodyClassifierApp:
    def __init__(self):
        self.rf_model = self.load_rf_model()
        self.recommendation_images = {
            "Female": {
                "APPLE": {
                    "Skirt": ["images\Women/a line skirt .png", "images\Women/wrap skirt .png", "images\Women/handkerchief skirt .png", "images\Women/flip skirt .png", "images\Women/draped skirt .png"],
                    "Jumpsuits": ["images\Women/belted jumpsuit .png", "images\Women/wide leg jumpsuit .png", "images\Women/utility jumpsuit .png", "images\Women/wrap jumpsuit .png", "images\Women/empire jumpsuit .png"],
                    "Pants": ["images\Women/harem pants .png", "images\Women/bootcut pants.png", "images\Women/Palazzo pants .png", "images\Women/pegged pants.png", "images\Women/wideleg jeans pants.png"],
                    "Necklines": ["images\Women/y neckline .png", "images\Women/v neckline.png", "images\Women/sweetheart neckline .png", "images\Women/scoop neckline .png", "images\Women/off shoulder neckline .png"],
                    "Tops": ["images\Women/off shoulder top .png", "images\Women/peplum top .png", "images\Women/wrap top.png", "images\Women/empire top.png", "images\Women/hoodie - top.png"],
                    "Sleeves": ["images\Women/cap sleeve .png", "images\Women/Bell sleeve.png", "images\Women/dolman sleeve.png", "images\Women/flutter sleeve .png", "images\Women/off shoulder sleeve .png"],
                    "TRADITIONAL WEAR": ["images\Women/aline kurta.png", "images\Women/anarkali kurta.png", "images\Women/straight cut kurta.png", "images\Women/empire waist kurta.png", "images\Women/saree.png"]
                }
            },
            "Male": {}
        }

    def load_rf_model(self):
        try:
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the RandomForestClassifier model: {e}")
            return None
        return None

    def classify(self, gender, age, measurements):
        try:
            # Get user inputs
            gender = self.gender_var.get()
            age = float(self.age_entry.get())
            measurements = [float(entry.get()) for entry in self.measurement_entries]

            # Create DataFrame from user inputs
            data = pd.DataFrame(columns=['Gender', 'Age', 'Shoulder', 'Waist', 'Hips', 'Bust', 'Chest'])
            data.loc[0] = [FEMALE_GENDER if gender == "Female" else MALE_GENDER, age] + measurements

            # Predict body type using RandomForestClassifier
            if self.rf_model:
                body_type = self.rf_model.predict(data)
                self.result_label.config(text=f"Predicted Body Type: {body_type[0]}")

                # Display recommendations based on predicted body type
                self.display_recommendations(gender, body_type[0])
            else:
                messagebox.showerror("Error", "RandomForestClassifier Model not loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

        

    def display_recommendations(self, gender, body_type):
         # Clear any previous recommendations
        if hasattr(self, 'recommendation_frames'):
            for frame in self.recommendation_frames:
                frame.destroy()

        # Create recommendation frames
        self.recommendation_frames = []
        row = len(self.measurement_labels) + 5
        col = 0
        for cloth_pattern, image_paths in self.recommendation_images[gender][body_type].items():
            tk.Label(self.master, text=f"Top images for {cloth_pattern}:").grid(row=row, column=col, padx=5, pady=5)
            row += 1
            frame = tk.Frame(self.master)
            frame.grid(row=row, column=col, padx=5, pady=5)
            self.recommendation_frames.append(frame)
            for i, image_path in enumerate(image_paths):
                # Use absolute path and os.path.join()
                full_image_path = os.path.join(os.getcwd(), image_path)
                img = Image.open(full_image_path)
                img = img.resize((100, 100), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                img_label = tk.Label(frame, image=img)
                img_label.image = img
                img_label.grid(row=i, column=0, padx=5, pady=5)
            row = len(self.measurement_labels) + 5
            col += 1

def main():
    st.title("Body Measurement Classifier")

    # Gender selection
    gender = st.selectbox("Gender:", ["Female", "Male"])

    # Age input
    age = st.number_input("Age:")

    # Measurement inputs
    measurement_labels = ["Shoulder", "Waist", "Hips", "Bust", "Chest"]
    measurements = [st.number_input(label) for label in measurement_labels]

    # Classify button
    app = BodyClassifierApp()
    if st.button("Classify"):
        app.classify(gender, age, measurements)
        # Assuming classification will trigger recommendation display
        app.display_recommendations(gender, "APPLE")

if __name__ == "__main__":
    main()

