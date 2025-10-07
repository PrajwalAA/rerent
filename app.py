# rental_property_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set visualization styles
sns.set(style='whitegrid')

# --- 1. App Title ---
st.title("🏠 Rental Property Data Analysis - India")

# --- 2. Upload Dataset ---
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Preview of dataset:", df.head())

    # --- 3. Data Cleaning ---
    st.header("🔹 Data Cleaning")
    # Strip column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)
    
    # Missing Values
    if st.checkbox("Show Missing Values"):
        st.write(df.isnull().sum())
    
    fill_option = st.selectbox("Fill missing values with:", ["0", "-1", "Custom"])
    if fill_option == "0":
        df.fillna(0, inplace=True)
    elif fill_option == "-1":
        df.fillna(-1, inplace=True)
    elif fill_option == "Custom":
        st.write("You can fill missing values manually in preprocessing step")
    
    # Remove Duplicates
    if st.checkbox("Remove Duplicate Rows"):
        df = df.drop_duplicates()
        st.success("Duplicates removed!")

    st.write("Cleaned Dataset Preview:", df.head())

    # --- 4. Feature Engineering ---
    st.header("🔹 Feature Engineering")
    # Categorize Property Age
    conditions_age = [
        (df['property_age_yrs'] <= 5),
        (df['property_age_yrs'] >= 6) & (df['property_age_yrs'] < 10),
        (df['property_age_yrs'] >= 10)
    ]
    labels_age = ['New', 'Mid-age', 'Old']
    df['age_category'] = np.select(conditions_age, labels_age)

    # Rent Segment
    conditions_rent = [
        (df['rent_month'] < 15000),
        (df['rent_month'] >= 15000) & (df['rent_month'] < 40000),
        (df['rent_month'] >= 40000)
    ]
    labels_rent = ['Affordable', 'Mid-range', 'Luxury']
    df['rent_segment'] = np.select(conditions_rent, labels_rent)

    # Rent per sqft
    df['rent_per_sqft'] = df.apply(
        lambda row: row['rent_month'] / row['size_sqft'] if pd.notnull(row['rent_month']) and pd.notnull(row['size_sqft']) and row['size_sqft'] != 0 else np.nan,
        axis=1
    ).round(2)

    # City Tier Mapping
    tier_map = {
        'mumbai': 'Tier-1', 'delhi': 'Tier-1', 'bangalore': 'Tier-1',
        'hyderabad': 'Tier-1', 'chennai': 'Tier-1', 'kolkata': 'Tier-1',
        'pune': 'Tier-2', 'nagpur': 'Tier-2', 'indore': 'Tier-2', 'jaipur': 'Tier-2'
    }
    df['city_tier'] = df['city'].str.lower().map(tier_map).fillna('Other')

    st.write("Feature Engineering Done!", df.head())

    # --- 5. Encoding ---
    st.header("🔹 Encoding")
    le = LabelEncoder()
    df['furnishing_encoded'] = le.fit_transform(df['furnishing'])
    df['propertytype_encoded'] = le.fit_transform(df['property_type'])
    df = pd.get_dummies(df, columns=['city'], prefix='city', drop_first=False)
    st.write("Encoding Completed!", df.head())

    # --- 6. Visualization ---
    st.header("📊 Visualizations")

    st.subheader("Rent Distribution")
    plt.figure(figsize=(10, 6))
    plt.hist(df['rent_month'], bins=30, color='skyblue', edgecolor='black')
    plt.axvline(15000, color='green', linestyle='--', label='Affordable/Mid-range')
    plt.axvline(40000, color='red', linestyle='--', label='Mid-range/Luxury')
    plt.xlabel('Rent Amount (₹)')
    plt.ylabel('Number of Properties')
    plt.title('Rent Distribution')
    plt.legend()
    st.pyplot(plt.gcf())  # Use plt.gcf() to get current figure
    plt.clf()  # Clear the figure after plotting

    st.subheader("Average Rent by Furnishing Type")
    plt.figure(figsize=(10, 6))
    furnish_avg = df.groupby('furnishing')['rent_month'].mean().sort_values()
    sns.barplot(x=furnish_avg.index, y=furnish_avg.values, palette='viridis')
    plt.ylabel('Average Rent (₹)')
    plt.xlabel('Furnishing Type')
    plt.title("Average Rent by Furnishing Type")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Average Rent by City Tier")
    plt.figure(figsize=(10, 6))
    tier_avg = df.groupby('city_tier')['rent_month'].mean().sort_values()
    sns.barplot(x=tier_avg.index, y=tier_avg.values, palette='magma')
    plt.ylabel('Average Rent (₹)')
    plt.xlabel('City Tier')
    plt.title("Average Rent by City Tier")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    corr_cols = ['rent_month', 'bhk', 'size_sqft', 'property_age_yrs']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap: Rent, BHK, Size, Age')
    st.pyplot(plt.gcf())
    plt.clf()

    # --- 7. Download cleaned dataset ---
    st.header("💾 Download Cleaned Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=csv,
        file_name='cleaned_rental_data.csv',
        mime='text/csv',
    )
