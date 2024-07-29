import streamlit as st
from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models, pull

st.title("AutoML with PyCaret and Streamlit")

# Load dataset
dataset = st.selectbox("Select Dataset", ("iris", "wine", "breast_cancer"))
data = get_data(dataset)

st.write("Dataset Preview")
st.dataframe(data.head())

# Setting up the PyCaret environment
if st.button("Run AutoML"):
    with st.spinner("Training models..."):
        clf = setup(data, target=data.columns[-1], silent=True, html=False)
        best_model = compare_models()

        st.write("Best Model")
        st.write(best_model)

        results = pull()
        st.write("Model Comparison")
        st.dataframe(results)
