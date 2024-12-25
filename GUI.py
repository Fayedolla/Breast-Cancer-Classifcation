import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
import random
import os

# Function to load the selected model from a specified file path
def load_model(file_path):
    try:
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", f"Model file '{file_path}' does not exist!")
            return None

        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model
    except ImportError as e:
        messagebox.showerror("Error", f"ImportError: {e}")
        return None
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the model: {e}")
        return None

# Function to make predictions
def predict():
    model_name = model_var.get()
    if not model_name or model_name == "No models found":
        messagebox.showwarning("Warning", "Please select a model.")
        return

    try:
        # Parse input values from the input fields
        input_values = [float(entry.get()) for entry in input_fields]
        input_array = np.array([input_values])  # Convert to a 2D array

        # Get the file path for the selected model
        model_file_path = model_paths.get(model_name)
        if not model_file_path:
            messagebox.showerror("Error", "Model file path not found!")
            return

        # Load the model and make predictions
        model = load_model(model_file_path)
        if model:
            # Predicting using the model
            prediction = model.predict(input_array)

            # If you are using a classifier, you may need to display the predicted class
            # Assuming this is a binary classifier, adjust accordingly
            prediction_label = prediction[0]  # If binary classification, prediction is usually a single value
            messagebox.showinfo("Prediction", f"The predicted class is: {prediction_label}")
        else:
            messagebox.showerror("Error", "Failed to load model.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to insert random numbers into the input fields
def insert_random_values():
    for entry in input_fields:
        random_value = round(random.uniform(0, 3000), 2)  # Generate random values between 0 and 30
        entry.delete(0, tk.END)  # Clear the existing value
        entry.insert(0, str(random_value))  # Insert random value

# Define paths for each model
model_paths = {
    "KNN": r"D:\University\Level - 3\Semester - 1\Machine Learning\Project\KNN",
    "Logistic_Regression": r"D:\University\Level - 3\Semester - 1\Machine Learning\Project\Logistic_Regression",
    "SVM": r"D:\University\Level - 3\Semester - 1\Machine Learning\Project\SVM",
    "Naive_Bayes": r"D:\University\Level - 3\Semester - 1\Machine Learning\Project\Naive_Bayes",
    "Decision_Tree": r"D:\University\Level - 3\Semester - 1\Machine Learning\Project\Decision_Tree"
}

# GUI setup
root = tk.Tk()  
root.title("Model Selection and Prediction GUI")
root.geometry("700x850")  # Adjusted window size

# Apply ttk theme for modern appearance
style = ttk.Style()
style.theme_use('clam')  # A modern theme

# Styling with more control
style.configure('TLabel', font=('Arial', 10, 'bold'), background="#1c1c1c", foreground="white")
style.configure('TEntry', font=('Arial', 10), fieldbackground="#333333", foreground="white", padding=5)
style.configure('TButton', font=('Arial', 12, 'bold'), background="#333333", foreground="white", padding=10)
style.map('TButton', background=[('active', '#444444')], foreground=[('active', 'white')])  # Hover effect

# Set black background for the root window
root.configure(bg="#1c1c1c")  # Dark black background

label_font = ('Arial', 10, 'bold')
entry_font = ('Arial', 10)

# Header
header_label = tk.Label(root, text="Model Selection and Prediction", font=('Arial', 18, 'bold'), bg="#000000", fg="white", pady=15)
header_label.pack(fill=tk.X, pady=10)

# Dropdown menu for model selection
model_var = tk.StringVar()
models = list(model_paths.keys())

# If no models are found, display a default option
if not models:
    models = ["No models found"]

# Label and dropdown for model selection
select_model_label = ttk.Label(root, text="Select Model:")
select_model_label.pack(pady=5)

model_menu = ttk.Combobox(root, textvariable=model_var, values=models, state="readonly", font=entry_font, width=30)
model_menu.pack(pady=10)

# Input fields for prediction values based on dataset features
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Create a frame for input fields with a scrollbar
input_frame = tk.Frame(root, bg="#1c1c1c")
input_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# Add a canvas and scrollbar for the input frame
canvas = tk.Canvas(input_frame, bg="#1c1c1c", highlightthickness=0)  # Ensuring background matches
scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#1c1c1c")  # Use tk.Frame to allow background setting

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.config(yscrollcommand=scrollbar.set)

# Scrollbar
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# Display input fields dynamically with feature names
input_fields = []
for i, feature in enumerate(features):
    feature_label = tk.Label(scrollable_frame, text=f"{feature}:", bg="#1c1c1c", fg="white", font=label_font)
    feature_label.pack(fill=tk.X, padx=20, pady=5)  # Use pack for labels with padding
    entry = ttk.Entry(scrollable_frame, font=entry_font, width=25)
    entry.pack(fill=tk.X, padx=20, pady=5)  # Use pack for entry fields with padding
    input_fields.append(entry)

# Predict button
predict_button = ttk.Button(root, text="Predict", command=predict, style="TButton")
predict_button.pack(pady=20)

# Button to insert random values
random_button = ttk.Button(root, text="Insert Random Values", command=insert_random_values, style="TButton")
random_button.pack(pady=10)

root.mainloop()
