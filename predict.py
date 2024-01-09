import streamlit as st
import pickle
import numpy as np 
import sklearn

def load_model():

    with open('ai_webapp\saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
data = load_model()

regressor = data["model"]
country = data["country"]
education = data["education"]



def show_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    educations = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country_1 = st.selectbox("Country", countries)
    education_1 = st.selectbox("Education Level", educations)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country_1, education_1, expericence]])
        X[:, 0] = country.fit_transform(X[:,0])
        X[:, 1] = education.fit_transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f'The estimate salary in USD(Annualy) is : ${salary[0]:.2f}')