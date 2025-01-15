import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#link Font 
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300..700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Text:ital@0;1&display=swap');
        * {
            font-family: 'Quicksand', sans-serif;
            text-align: center;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset dan tampilkan judul
st.markdown(
    """
    <h1 style='text-align: center; font-family: "DM Serif Text", serif;'>
        <img src="https://img.icons8.com/emoji/96/000000/robot-emoji.png" width="50" style="vertical-align: middle;"> 
        Aplikasi Regresi dengan Pilihan Algoritma
    </h1>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center; font-size:18px;'>
        Aplikasi ini membantu memprediksi biaya medis berdasarkan dataset historis.
    </p>
    """, 
    unsafe_allow_html=True
)


data = pd.read_csv('Regression.csv')

# Handle categorical data
categorical_columns = data.select_dtypes(include=['object']).columns
encoders = {}
if not categorical_columns.empty:
    st.info("Data memiliki kolom kategori yang akan dienkode menjadi numerik.")
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

st.write("### Dataset")
st.dataframe(data)

# Feature Selection
st.sidebar.markdown(
    """
    <h3 style="display: flex; align-items: center; margin-bottom: 10px;">
        <img src="https://img.icons8.com/small/16/input-component.png" width="24" height="24 alt="input-icon" style="margin-right: 8px; filter: invert(100%)"/>
        Input Features
    </h3>
    """,
    unsafe_allow_html=True
)
target_column = st.sidebar.selectbox("Pilih Kolom Target", data.columns)
feature_columns = st.sidebar.multiselect("Pilih Kolom Fitur", [col for col in data.columns if col != target_column])

if target_column and feature_columns:
    X = data[feature_columns]
    y = data[target_column]

    # Splitting the data
    test_size = st.sidebar.slider("Ukuran Data Uji (%)", 10, 50, 20) / 100
    random_state = st.sidebar.slider("Random State", 0, 100, 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Algorithm Selection
    algorithm = st.sidebar.selectbox("Pilih Algoritma Regresi", [
        "Decision Tree Regression",
        "Random Forest Regression",
        "Multiple Linear Regression",
        "Simple Linear Regression",
        "Non-linear Regression",
        "Support Vector Regression"
    ])

    # Train and Evaluate Model
    st.write(f"### Hasil dengan Algoritma: {algorithm}")
    if algorithm == "Decision Tree Regression":
        model = DecisionTreeRegressor(random_state=random_state)
    elif algorithm == "Random Forest Regression":
        n_estimators = st.sidebar.slider("Jumlah Estimator (Trees)", 10, 200, 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    elif algorithm == "Multiple Linear Regression":
        model = LinearRegression()
    elif algorithm == "Simple Linear Regression":
        if len(feature_columns) > 1:
            st.warning("Regresi Linear Sederhana hanya mendukung satu fitur. Pilih satu kolom fitur saja.")
            st.stop()
        model = LinearRegression()
    elif algorithm == "Non-linear Regression":
        degree = st.sidebar.slider("Degree Polinomial", 2, 5, 3)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif algorithm == "Support Vector Regression":
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = SVR(kernel=kernel)

    # Train model
    model.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("#### Evaluasi Model")
    predictions = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
    })

    st.write("### Hasil Prediksi pada Dataset Uji")
    st.dataframe(predictions)
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"R-Squared (R2): {r2:.4f}")

    # Plot Predictions vs Actual
    st.write("#### Visualisasi Hasil")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    st.pyplot(fig)

    # User Input for New Prediction
    st.write("### Prediksi Biaya Medis Baru")
    with st.form("input_form"):
        age = st.number_input("Usia", min_value=1, max_value=120, value=30)
        sex = st.selectbox("Jenis Kelamin", encoders["sex"].classes_)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Perokok", encoders["smoker"].classes_)
        region = st.selectbox("Region", encoders["region"].classes_)
        submit = st.form_submit_button("Prediksi")

    if submit:
        # Encode user input
        user_input = pd.DataFrame({
    'age': [age],
    'sex': [encoders["sex"].transform([sex])[0]],
    'bmi': [bmi],
    'children': [children],
    'smoker': [encoders["smoker"].transform([smoker])[0]],
    'region': [encoders["region"].transform([region])[0]]
}, columns=feature_columns)  # Pastikan kolom sesuai dengan feature_columns

        
        # Predict charges
        predicted_charges = model.predict(user_input)[0]
        st.write(f"### Prediksi Biaya Medis: ${predicted_charges:.2f}")

else:
    st.warning("Pilih kolom target dan setidaknya satu kolom fitur.")


st.write("---")
st.caption("Dibuat oleh Putu Puja Diva Widiasari NIM 211220040")

