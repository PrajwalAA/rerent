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
    
    # Categorize Property Age - with error handling
    if 'property_age_yrs' in df.columns:
        try:
            # Ensure column is numeric
            df['property_age_yrs'] = pd.to_numeric(df['property_age_yrs'], errors='coerce')
            
            conditions_age = [
                (df['property_age_yrs'] <= 5),
                (df['property_age_yrs'] >= 6) & (df['property_age_yrs'] < 10),
                (df['property_age_yrs'] >= 10)
            ]
            labels_age = ['New', 'Mid-age', 'Old']
            df['age_category'] = np.select(conditions_age, labels_age, default='Unknown')
        except Exception as e:
            st.error(f"Error creating age categories: {e}")
            df['age_category'] = 'Unknown'
    else:
        st.warning("'property_age_yrs' column not found. Skipping age categorization.")
        df['age_category'] = 'Unknown'

    # Rent Segment - with error handling
    if 'rent_month' in df.columns:
        try:
            # Ensure column is numeric
            df['rent_month'] = pd.to_numeric(df['rent_month'], errors='coerce')
            
            conditions_rent = [
                (df['rent_month'] < 15000),
                (df['rent_month'] >= 15000) & (df['rent_month'] < 40000),
                (df['rent_month'] >= 40000)
            ]
            labels_rent = ['Affordable', 'Mid-range', 'Luxury']
            df['rent_segment'] = np.select(conditions_rent, labels_rent, default='Unknown')
        except Exception as e:
            st.error(f"Error creating rent segments: {e}")
            df['rent_segment'] = 'Unknown'
    else:
        st.warning("'rent_month' column not found. Skipping rent segmentation.")
        df['rent_segment'] = 'Unknown'

    # Rent per sqft - with error handling
    if 'rent_month' in df.columns and 'size_sqft' in df.columns:
        try:
            # Ensure columns are numeric
            df['rent_month'] = pd.to_numeric(df['rent_month'], errors='coerce')
            df['size_sqft'] = pd.to_numeric(df['size_sqft'], errors='coerce')
            
            df['rent_per_sqft'] = df.apply(
                lambda row: row['rent_month'] / row['size_sqft'] 
                if pd.notnull(row['rent_month']) and pd.notnull(row['size_sqft']) and row['size_sqft'] != 0 
                else np.nan,
                axis=1
            ).round(2)
        except Exception as e:
            st.error(f"Error calculating rent per sqft: {e}")
            df['rent_per_sqft'] = np.nan
    else:
        st.warning("Required columns for rent per sqft calculation not found. Skipping this step.")
        df['rent_per_sqft'] = np.nan

    # City Tier Mapping - with error handling
    if 'city' in df.columns:
        try:
            tier_map = {
                'mumbai': 'Tier-1', 'delhi': 'Tier-1', 'bangalore': 'Tier-1',
                'hyderabad': 'Tier-1', 'chennai': 'Tier-1', 'kolkata': 'Tier-1',
                'pune': 'Tier-2', 'nagpur': 'Tier-2', 'indore': 'Tier-2', 'jaipur': 'Tier-2'
            }
            df['city_tier'] = df['city'].str.lower().map(tier_map).fillna('Other')
        except Exception as e:
            st.error(f"Error mapping city tiers: {e}")
            df['city_tier'] = 'Other'
    else:
        st.warning("'city' column not found. Skipping city tier mapping.")
        df['city_tier'] = 'Other'

    # --- NEW: Date Feature Engineering ---
    if 'availability_date' in df.columns:
        try:
            # Convert to datetime
            df['availability_date'] = pd.to_datetime(df['availability_date'], errors='coerce')
            
            # Create new features
            df['year'] = df['availability_date'].dt.year
            df['month'] = df['availability_date'].dt.month
            df['day_of_week'] = df['availability_date'].dt.day_name()
            
            # Drop the original date column
            df = df.drop('availability_date', axis=1)
            
            st.success("Date features created successfully!")
        except Exception as e:
            st.error(f"Error processing date features: {e}")
    else:
        st.warning("'availability_date' column not found. Skipping date feature extraction.")

    st.write("Feature Engineering Done!", df.head())

    # --- 5. Encoding ---
    st.header("🔹 Encoding")
    le = LabelEncoder()
    
    # Encode furnishing with error handling
    if 'furnishing' in df.columns:
        try:
            df['furnishing_encoded'] = le.fit_transform(df['furnishing'].astype(str))
        except Exception as e:
            st.error(f"Error encoding furnishing: {e}")
            df['furnishing_encoded'] = -1
    else:
        st.warning("'furnishing' column not found. Skipping encoding.")
        df['furnishing_encoded'] = -1
    
    # Encode property type with error handling
    if 'property_type' in df.columns:
        try:
            df['propertytype_encoded'] = le.fit_transform(df['property_type'].astype(str))
        except Exception as e:
            st.error(f"Error encoding property type: {e}")
            df['propertytype_encoded'] = -1
    else:
        st.warning("'property_type' column not found. Skipping encoding.")
        df['propertytype_encoded'] = -1
    
    # One-hot encode cities with error handling
    if 'city' in df.columns:
        try:
            df = pd.get_dummies(df, columns=['city'], prefix='city', drop_first=False)
        except Exception as e:
            st.error(f"Error one-hot encoding cities: {e}")
    else:
        st.warning("'city' column not found. Skipping one-hot encoding.")
    
    st.write("Encoding Completed!", df.head())

    # --- 6. Visualization ---
    st.header("📊 Visualizations")

    # Rent Distribution - with error handling
    if 'rent_month' in df.columns:
        try:
            st.subheader("Rent Distribution")
            plt.figure(figsize=(10, 6))
            plt.hist(df['rent_month'], bins=30, color='skyblue', edgecolor='black')
            plt.axvline(15000, color='green', linestyle='--', label='Affordable/Mid-range')
            plt.axvline(40000, color='red', linestyle='--', label='Mid-range/Luxury')
            plt.xlabel('Rent Amount (₹)')
            plt.ylabel('Number of Properties')
            plt.title('Rent Distribution')
            plt.legend()
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error creating rent distribution plot: {e}")

    # Average Rent by Furnishing Type - with error handling
    if 'furnishing' in df.columns and 'rent_month' in df.columns:
        try:
            st.subheader("Average Rent by Furnishing Type")
            plt.figure(figsize=(10, 6))
            furnish_avg = df.groupby('furnishing')['rent_month'].mean().sort_values()
            sns.barplot(x=furnish_avg.index, y=furnish_avg.values, palette='viridis')
            plt.ylabel('Average Rent (₹)')
            plt.xlabel('Furnishing Type')
            plt.title("Average Rent by Furnishing Type")
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error creating furnishing rent plot: {e}")

    # Average Rent by City Tier - with error handling
    if 'city_tier' in df.columns and 'rent_month' in df.columns:
        try:
            st.subheader("Average Rent by City Tier")
            plt.figure(figsize=(10, 6))
            tier_avg = df.groupby('city_tier')['rent_month'].mean().sort_values()
            sns.barplot(x=tier_avg.index, y=tier_avg.values, palette='magma')
            plt.ylabel('Average Rent (₹)')
            plt.xlabel('City Tier')
            plt.title("Average Rent by City Tier")
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error creating city tier rent plot: {e}")

    # NEW: Visualizations for Date Features
    if 'month' in df.columns:
        try:
            st.subheader("Property Availability by Month")
            plt.figure(figsize=(10, 6))
            month_counts = df['month'].value_counts().sort_index()
            sns.barplot(x=month_counts.index, y=month_counts.values, palette='viridis')
            plt.xlabel('Month')
            plt.ylabel('Number of Properties')
            plt.title('Property Availability by Month')
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error creating month availability plot: {e}")

    if 'day_of_week' in df.columns:
        try:
            st.subheader("Property Availability by Day of Week")
            plt.figure(figsize=(10, 6))
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['day_of_week'].value_counts().reindex(day_order).dropna()
            sns.barplot(x=day_counts.index, y=day_counts.values, palette='magma')
            plt.xlabel('Day of Week')
            plt.ylabel('Number of Properties')
            plt.title('Property Availability by Day of Week')
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error creating day of week availability plot: {e}")

    # Correlation Heatmap - with error handling
    corr_cols = ['rent_month', 'bhk', 'size_sqft', 'property_age_yrs']
    if all(col in df.columns for col in corr_cols):
        try:
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(10, 8))
            corr_matrix = df[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlation Heatmap: Rent, BHK, Size, Age')
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")
    else:
        missing_cols = [col for col in corr_cols if col not in df.columns]
        st.warning(f"Cannot create correlation heatmap. Missing columns: {missing_cols}")

    # --- 7. Download cleaned dataset ---
    st.header("💾 Download Cleaned Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=csv,
        file_name='cleaned_rental_data.csv',
        mime='text/csv',
    )
