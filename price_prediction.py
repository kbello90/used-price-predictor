import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Changed from pickle to joblib
from io import BytesIO

# Load the model - now using joblib
@st.cache_resource
def load_model():
    with open('used_price_model_kb.pkl', 'rb') as f:
        model = joblib.load(f)  # Changed to joblib.load
    return model

model = load_model()

# Brand mapping
brand_options = {
    'Other': 0,
    'Lenovo': 1,
    'Nokia': 2,
    'Xiaomi': 3
}

# App layout
st.title('📱 Used Phone Price Predictor by Karen Bello')
st.markdown("""
Predict the market price of your used smartphone based on its specifications.
This model considers various factors including hardware specs, brand, and age of the device.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Phone Specifications")
    
    brand = st.selectbox(
        "Brand",
        options=list(brand_options.keys()),
        index=0
    )
    
    screen_size = st.slider(
        "Screen Size (inches)",
        min_value=4.0,
        max_value=7.5,
        value=6.1,
        step=0.1
    )
    
    main_camera = st.slider(
        "Main Camera (MP)",
        min_value=2,
        max_value=200,
        value=12,
        step=1
    )
    
    selfie_camera = st.slider(
        "Selfie Camera (MP)",
        min_value=2,
        max_value=100,
        value=8,
        step=1
    )
    
    ram = st.slider(
        "RAM (GB)",
        min_value=1,
        max_value=16,
        value=4,
        step=1
    )
    
    has_4g = st.checkbox(
        "4G Supported",
        value=True
    )
    
    days_used = st.slider(
        "Days Used",
        min_value=1,
        max_value=365*5,
        value=365,
        step=1
    )
    
    original_price = st.number_input(
        "Original Price (USD)",
        min_value=50,
        max_value=2000,
        value=500,
        step=50
    )
    
    years_since_release = st.slider(
        "Years Since Release",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

# Calculate normalized price
normalized_new_price = original_price / 1000

# Create input dataframe
input_data = {
    'const': [1],
    'screen_size': [screen_size],
    'main_camera_mp': [main_camera],
    'selfie_camera_mp': [selfie_camera],
    'ram': [ram],
    'days_used': [days_used],
    'normalized_new_price': [normalized_new_price],
    'years_since_release': [years_since_release],
    'brand_name_Lenovo': [1 if brand == 'Lenovo' else 0],
    'brand_name_Nokia': [1 if brand == 'Nokia' else 0],
    'brand_name_Xiaomi': [1 if brand == 'Xiaomi' else 0],
    '4g_yes': [1 if has_4g else 0]
}

input_df = pd.DataFrame(input_data)

# Ensure correct column order
expected_columns = [
    'const',
    'screen_size',
    'main_camera_mp',
    'selfie_camera_mp',
    'ram',
    'days_used',
    'normalized_new_price',
    'years_since_release',
    'brand_name_Lenovo',
    'brand_name_Nokia',
    'brand_name_Xiaomi',
    '4g_yes'
]

input_df = input_df[expected_columns]

# Prediction button
if st.button('Predict Price'):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        predicted_price = prediction[0] * 1000
        
        # Display result
        st.success(f"Predicted Used Price: ${predicted_price:,.2f}")
        
        # Calculate depreciation
        depreciation = original_price - predicted_price
        depreciation_pct = (depreciation / original_price) * 100
        
        st.metric(
            label="Depreciation",
            value=f"${depreciation:,.2f}",
            delta=f"{depreciation_pct:.1f}% of original price"
        )
        
        # Show feature importance if available
        try:
            if hasattr(model, 'params'):
                st.subheader("Feature Impact on Price")
                coefficients = model.params.drop('const')
                coefficients = coefficients.sort_values(ascending=False)
                st.bar_chart(coefficients)
        except Exception as e:
            st.warning(f"Couldn't display feature importance: {str(e)}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Explanations
with st.expander("How this prediction works"):
    st.markdown("""
    This price predictor uses a machine learning model trained on historical data of used phone sales. 
    The model considers:
    
    - **Hardware specifications**: Screen size, camera quality, RAM
    - **Usage**: How long the phone has been used
    - **Age**: Years since the phone was first released
    - **Brand**: Manufacturer's impact on resale value
    - **Network**: Whether the phone supports 4G
    
    The prediction is an estimate based on market trends and may vary based on condition, location, and other factors.
    """)

# Download button
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(input_df.drop(columns=['const']))

st.download_button(
    label="Download Input Data as CSV",
    data=csv,
    file_name='phone_specs.csv',
    mime='text/csv'
)
