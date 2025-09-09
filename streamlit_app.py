import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import os

# To define the dataset 
try:
    df = pd.read_csv("cleaned_price_dataset.csv")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Dropping of any unnamed columns and stripping the whitespace from column names
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

# Fix the column name issue (ï»¿laptop_ID is likely a BOM issue)
if df.columns[0].startswith('ï»¿'):
    df = df.rename(columns={df.columns[0]: "laptop_ID"})

# Creating and Training model 
x = df.drop(columns=["Price_euros", "laptop_ID", "Inches", "Weight", "ScreenResolution"], axis=1) 
y = df["Price_euros"]  # target variable

x_train, x_test, y_train, y_test = split(x, y, test_size=0.2, random_state=5)

# Identify the categorical and the numerical columns
categorical_columns = x.select_dtypes(include=['object']).columns
numerical_columns = x.select_dtypes(exclude=['object']).columns

# Transforming Encode categorical + scale numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('category', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('numbers', StandardScaler(), numerical_columns)
    ]
)

# Pipeline: the preprocessing and the modelling 
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# the train 
model.fit(x_train, y_train)

# The function to predict laptop price of the user
def get_price(user_input):
    """
    Predicts the laptop price based on user input.
    The function creates a DataFrame from the input and passes it directly
    to the trained pipeline for prediction.
    """
    # Creating the DataFrame from the user's input list, using the original
    # training columns for the headers.
    user_data_df = pd.DataFrame([user_input], columns=x.columns)
  
    # The pipeline will handle both the preprocessing
    # and the prediction in one step.
    predicted_price = model.predict(user_data_df)
  
    return predicted_price

# Helper function to safely get unique values for dropdowns
def get_unique_sorted_values(column):
    try:
        # Convert to string to handle any data type issues
        unique_vals = column.astype(str).unique()
        # Remove any NaN values
        unique_vals = [val for val in unique_vals if val != 'nan' and pd.notna(val)]
        return sorted(unique_vals)
    except:
        return []


st.title("Group A2 Laptop Price Prediction Project")
st.write("Choose your desired Laptop features to predict the price")

# Creating a selection box to get user input
Company = st.selectbox("Company", get_unique_sorted_values(x["Company"]))
Product = st.selectbox("Product", get_unique_sorted_values(x["Product"]))
TypeName = st.selectbox("Type", get_unique_sorted_values(x["TypeName"]))
Cpu = st.selectbox("CPU", get_unique_sorted_values(x["Cpu"]))
Ram = st.selectbox("RAM", get_unique_sorted_values(x["Ram"]))
Memory = st.selectbox("Memory", get_unique_sorted_values(x["Memory"]))
Gpu = st.selectbox("GPU", get_unique_sorted_values(x["Gpu"]))
OpSys = st.selectbox("Operating System", get_unique_sorted_values(x["OpSys"]))

user_input = [Company, Product, TypeName, Cpu, Ram, Memory, Gpu, OpSys]

# shows the price of the device
if st.button("Predict price"):
    try:
        predicted_price = get_price(user_input)[0]
        # Checks for the negative price predictions and handle them
        if predicted_price < 0:
            st.write("The predicted price appears to be negative, which may indicate an issue with the input data.")
        else:
            text = f"The estimated price is €{predicted_price:,.2f}"
            st.write(text)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
