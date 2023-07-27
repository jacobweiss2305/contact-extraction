from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd


from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain

json_schema = {
    "title": "Person",
    "description": "Identifying information about a person.",
    "type": "object",
    "properties": {
        "first_name": {"title": "First Name", "description": "The person's first name", "type": "string"},
        "last_name": {"title": "Last Name", "description": "The person's last name", "type": "string"},
        "job_title": {"title": "Job Title", "description": "The person's job title", "type": "string"},
        "company_name": {"title": "Company Name", "description": "The person's company name", "type": "string"},
        "mobile_number": {"title": "Mobile Number", "description": "The person's mobile number", "type": "string"},
        "office_number": {"title": "Office Number", "description": "The person's office number", "type": "string"},
        "business_website": {"title": "Business Website", "description": "The website of the company that the person works for", "type": "string"},
        "address": {"title": "Address", "description": "The address of the company that the person works for", "type": "string"},
    },
    "required": ["first_name", "last_name", "job_title", "company_name", "mobile_number", "office_number", "business_website", "email", "address"]
}

llm = ChatOpenAI(model="gpt-4", temperature=0)

def main():
    st.title("AI Contact Extraction App")
    st.write("Enter the conversation text:")
    conversation_input = st.text_area("Conversation", "")

    if st.button("Extract Contacts"):
            if conversation_input:

                chain = create_extraction_chain(json_schema, llm)

                output = chain.run(conversation_input)

                st.write(pd.DataFrame(output))

            else:
                st.warning("Please enter a conversation text.")

if __name__ == "__main__":
    main()
