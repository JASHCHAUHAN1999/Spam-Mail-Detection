import streamlit as st
import pickle as p

#python -m venv venv # for virtual enviroment
#pip install streamlit 
#
# streamlit run app.py --theme.base="dark" # to run code

with open(r'C:\Users\JASH\Desktop\MCA\Jupyter Notebook Prec\Spam Mail\Spam_Mail_Streamlit\SpamMail_LogisticRegression_model.pkl','rb') as m:
    ml = p.load(m)
with open(r'C:\Users\JASH\Desktop\MCA\Jupyter Notebook Prec\Spam Mail\Spam_Mail_Streamlit\Feature_TfidfVectorizer.pkl','rb') as f:
    ftr = p.load(f)


st.title("Spam Mail Prediction System...")

data = st.text_area("Paste your mail contain here...",height=300)

button = st.button('Predict')


if button:
    if len(data)>0:
        f_data = ftr.transform([data])
        pred = ml.predict(f_data)
        if pred[0]==1:
            st.error("Alert it's Spam Mail...")
        else:
            st.success("Don't worry it's Ham Mail...")
    else:
        st.warning("Please enter your mail...")



