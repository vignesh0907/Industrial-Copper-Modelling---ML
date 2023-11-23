import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import re
import pickle

st.set_page_config(layout='wide')
st.title('Industrial Copper Modeling')
status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
         '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
         '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
         '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
         '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

model_selection = st.sidebar.selectbox('Select the Model', ('Regression - Predict the Selling Price', 'Classification - Predict the Status'))

if model_selection == 'Regression - Predict the Selling Price':
    col1,col2 = st.columns(2, gap='small')
    with col1:
        rstatus = st.selectbox("Status", status_options, index=None, placeholder='Select an Option', key=1)
        ritem_type = st.selectbox("Item Type", item_type_options,index=None, placeholder='Select an Option', key=2)
        rcountry = st.selectbox("Country", sorted(country_options),index=None, placeholder='Select an Option', key=3)
        rapplication = st.selectbox("Application", sorted(application_options),index=None, placeholder='Select an Option', key=4)
        rproduct_ref = st.selectbox("Product Reference", product,index=None, placeholder='Select an Option', key=5)

    with col2:
        rquantity_tons = st.number_input("Enter Quantity Tons (Min:611728 & Max:1722207579)", min_value=611728, max_value=1722207579)
        rthickness = st.number_input("Enter thickness (Min:0.18 & Max:400)", value=0.18, min_value=float(0.18), max_value=float(400))
        rwidth = st.number_input("Enter width (Min:1, Max:2990)", min_value=1, max_value=2990)
        rcustomer = st.number_input("customer ID (Min:12458, Max:30408185)", min_value=12458, max_value=30408185)
        st.write('')
        st.write('')
        rsubmit_button = st.button("Predict Selling Price")

    flag = 0
    pattern = "^(?:\d+|\d*\.\d+)$"
    for i in [rquantity_tons,rthickness,rwidth,rcustomer]:
        if re.match(pattern, str(i)):
            pass
        else:                    
            flag = 1  
            break

    if rsubmit_button and flag==1:
        if len(i) == 0:
            st.write('Please Enter a Valid Value Spaces not allowed')
        else:
            st.write("Please Enter a Valid Value")

    if rsubmit_button and flag == 0:
        with open(r'model.pkl', 'rb') as file:
            executed_model = pickle.load(file)
        with open(r'scaler.pkl', 'rb') as f:
            scaler_executed = pickle.load(f)
        with open(r't.pkl', 'rb') as f:
            t_executed = pickle.load(f)
        with open(r's.pkl', 'rb') as f:
            s_executed = pickle.load(f)

        new_sample= np.array([[np.log(float(rquantity_tons)),rapplication,np.log(float(rthickness)),float(rwidth),rcountry,float(rcustomer),int(rproduct_ref),ritem_type,rstatus]])
        new_sample_ohe = t_executed.transform(new_sample[:, [7]]).toarray()
        new_sample_be = s_executed.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scaler_executed.transform(new_sample)
        new_pred = executed_model.predict(new_sample1)[0]
        st.write('## :green[Predicted Selling Price] ', np.exp(new_pred))


elif model_selection == 'Classification - Predict the Status':
    col3, col4 = st.columns(2, gap='small')
    with col3:
        cquantity_tons = st.number_input("Enter Quantity Tons (Min:611728 & Max:1722207579)", min_value=611728, max_value=1722207579)
        cthickness = st.number_input("Enter thickness (Min:0.18 & Max:400)", value=0.18, min_value=float(0.18), max_value=float(400))
        cwidth = st.number_input("Enter width (Min:1, Max:2990)", min_value=1, max_value=2990)
        ccustomer = st.number_input("customer ID (Min:12458, Max:30408185)", min_value=12458, max_value=30408185)
        cselling = st.number_input("Selling Price (Min:1, Max: 100001015)", min_value=1, max_value=100001015)
    with col4:
        citem_type = st.selectbox("Item Type", item_type_options,key=21)
        ccountry = st.selectbox("Country", sorted(country_options),key=31)
        capplication = st.selectbox("Application", sorted(application_options),key=41)  
        cproduct_ref = st.selectbox("Product Reference", product,key=51)  
        st.write('')
        st.write('')         
        csubmit_button = st.button(label="PREDICT STATUS")   

    
    cpattern = r"^(?:\d+|\d*\.\d+)$"
    cinput_values = [str(cquantity_tons), str(cthickness), str(cwidth), str(ccustomer), str(cselling)]
    cflag = 0
    for j in cinput_values:
        if re.match(cpattern, j):
            pass
        else:
            cflag = 1
            break
    if csubmit_button and cflag == 1:
        if len(j) == 0:
            st.write('Please Enter a Valid Value Spaces not allowed')
        else:
            st.write("Please Enter a Valid Value")

    if csubmit_button and cflag == 0:
        with open(r"cmodel.pkl", 'rb') as file:
            cloaded_model = pickle.load(file)
        with open(r'cscaler.pkl', 'rb') as f:
            cscaler_loaded = pickle.load(f)
        with open(r"ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)

        new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(cproduct_ref),citem_type]])
        new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
        new_sample = cscaler_loaded.transform(new_sample)
        new_pred = cloaded_model.predict(new_sample)

        if new_pred==1:
            st.write('## :green[The Status is Won]')
        else:
            st.write('## :red[The status is Lost]')