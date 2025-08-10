import streamlit as st 
from inference import predict_species,load_model

# load the model first.
model = load_model()

# initialise streamlit
st.set_page_config(page_title="Iris Project")

# Add title to the page
st.title("Iris Project")
st.subheader("By Sirishti Singh")

# take input from users
sep_len = st.number_input("Sepal length",min_value=0.00,step=0.01)
sep_wid = st.number_input("Sepal Width",min_value=0.00,step=0.01)
pet_len = st.number_input("Petal length",min_value=0.00,step=0.01)
pet_wid = st.number_input("Petal Width",min_value=0.00,step=0.01)

#create a button to predict results
button = st.button("predict",type="primary")
if button:
    preds,prob_df = predict_species(model,sep_len,sep_wid,pet_len,pet_wid)
    st.subheader(f"Prediction : {preds}")
    st.subheader("Probablities : ")
    st.dataframe(prob_df)
    st.bar_chart(prob_df.T)
    
