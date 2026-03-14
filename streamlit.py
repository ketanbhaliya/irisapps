import streamlit as st
st.title("Iris Prediction")
import pickle
model = pickle.load(open("model_svm.pkl", "rb"))
sl=st.slider("SL", 2.0, 10.0)
sw=st.slider("SW", 2.0, 10.0)
pl=st.slider("PL", 2.0, 10.0)
pw=st.slider("PW", 2.0, 10.0)
if st.button("predict"):
  st.success(model.predict([[sl,sw,pl,pw]]))