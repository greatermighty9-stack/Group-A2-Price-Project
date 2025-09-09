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


st.title("Group A2 Laptop Price Prediction Project")
st.write("Choose your desired Laptop features to predict the price")

# Creating a selection box to get user input
Company = st.selectbox("Company", sorted(list(set(x["Company"].tolist()))))
Product = st.selectbox("Product", sorted(list(set(x["Product"].tolist()))))
TypeName = st.selectbox("Type", sorted(list(set(x["TypeName"].tolist()))))
Cpu = st.selectbox("CPU", sorted(list(set(x["Cpu"].tolist()))))
Ram = st.selectbox("RAM", sorted(list(set(x["Ram"].tolist()))))
Memory = st.selectbox("Memory", sorted(list(set(x["Memory"].tolist()))))
Gpu = st.selectbox("GPU", sorted(list(set(x["Gpu"].tolist()))))
OpSys = st.selectbox("Operating System", sorted(list(set(x["OpSys"].tolist()))))

user_input = [Company, Product, TypeName, Cpu, Ram, Memory, Gpu, OpSys]

# shows the price of the device
if st.button("Predict price"):
    predicted_price = get_price(user_input)[0]
    # Checks for the negative price predictions and handle them
    if predicted_price < 0:
        st.write("The predicted price appears to be negative, which may indicate an issue with the input data.")
    else:
        text = f"The estimated price is €{predicted_price:,.2f}"
        st.write(text)
