import os
import streamlit as st
import utils
import os
from langchain.llms import OpenAI

llm = OpenAI()

st.title("Contact Extraction Bot")

selected_file  = st.selectbox("Select a file", os.listdir("emails"))

file_path = os.path.join("emails", selected_file)
text = utils.load_text_file(file_path)
st.text_area("File Content:", text, height=200)

prompt_text = """ 

Extract all contact information from the email.

Please format the contact information as follows:
FIRST NAME: John
LAST NAME: Doe
EMAIL:
PHONE:
COMPANY:

"""

st.text_area("Prompt:", prompt_text, height=200)

response = utils.qa(file_path, prompt_text)

st.text_area("Response:", response, height=200)