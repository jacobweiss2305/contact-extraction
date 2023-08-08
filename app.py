from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd

from pydantic import BaseModel, Field
from typing import Optional

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import create_pandas_dataframe_agent
from typing import List
from pydantic import BaseModel
import utils
import concurrent.futures

from typing import List, Optional

from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

import pandas as pd
from pydantic import BaseModel, Field, validator

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import re

from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm = Replicate(
    model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    # input={"temperature": 0.75, "max_length": 500, "top_p": 1},
)

st.set_page_config(layout="wide")

class ContactInfo:
    def __init__(self, first_name, last_name, job_title, company_name, mobile_number, office_number, business_website, email, address):
        self.first_name = first_name
        self.last_name = last_name
        self.job_title = job_title
        self.company_name = company_name
        self.mobile_number = mobile_number
        self.office_number = office_number
        self.business_website = business_website
        self.email = email
        self.address = address

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def to_lowercase(df):
    return df.applymap(lambda s: s.lower() if type(s) == str else s)

def get_unique_contacts_with_most_info(df):
    # Add a column that counts the number of non-null fields for each contact
    df['info_count'] = df.notnull().sum(axis=1)
    # Group by 'first_name', 'last_name', 'email', and get the row with max 'info_count' for each group
    df = df.loc[df.groupby(['first_name', 'last_name', ])['info_count'].idxmax()]
    # Drop 'info_count' as it's no longer needed
    df = df.drop(columns='info_count')
    return df

def split_text_by_from(text):
    # Split the text by "From:" and strip any leading/trailing spaces from each part
    split_text = [part.strip() for part in text.split("From:")]

    # Remove the empty string at the beginning caused by the split
    split_text = [part for part in split_text if part]

    return split_text

def process_email(chain, email):
    output = chain.run(email)
    return pd.DataFrame(output.dict()['personal_info'])

def create_prompt(email):
    prompt = f"""

        System: As a world-class algorithm for extracting information in structured formats, your task is to extract the contact details from the given email's signature and cc section. The email is provided below:

        Human:
        {email}

        Human:
        Extract the contact information from the email signature and cc section.

        The information should be accurately extracted and arranged in a way that follows the correct format:

        Person 1
        First Name: 
        Last Name:
        Job Title:
        Email:
        Mobile Phone Number: 
        Office Phone Number: 
        Company Name: 
        Company Website:
        Company Address:

        Person 2
        First Name: 
        Last Name:
        Job Title:
        Email:
        Mobile Phone Number: 
        Office Phone Number: 
        Company Name: 
        Company Website:
        Company Address:

        Person N
        First Name: 
        Last Name:
        Job Title:
        Email:
        Mobile Phone Number: 
        Office Phone Number: 
        Company Name: 
        Company Website:
        Company Address:

        """
    return prompt

def parse_contact_info(input_list):
    personal_info_list = []
    entry_pattern = re.compile(
        r'\s*First Name:\s*(?P<first_name>[^\n]*)\s*'
        r'Last Name:\s*(?P<last_name>[^\n]*)\s*'
        r'Job Title:\s*(?P<job_title>[^\n]*)\s*'
        r'Email:\s*(?P<email>[^\n]*)\s*'
        r'Mobile Phone Number:\s*(?P<mobile_number>[^\n]*)\s*'
        r'Office Phone Number:\s*(?P<office_number>[^\n]*)\s*'
        r'Company Name:\s*(?P<company_name>[^\n]*)\s*'
        r'Company Website:\s*(?P<business_website>[^\n]*)\s*'
        r'Company Address:\s*(?P<address>[^\n]*)'
    )

    for entry in input_list:
        # strip leading/trailing whitespaces from the entry string
        entry = entry.strip()
        match = entry_pattern.search(entry)
        if match:
            contact_info = ContactInfo(
                first_name=match.group('first_name'),
                last_name=match.group('last_name'),
                job_title=match.group('job_title'),
                company_name=match.group('company_name'),
                mobile_number=match.group('mobile_number'),
                office_number=match.group('office_number'),
                business_website=match.group('business_website'),
                email=match.group('email'),
                address=match.group('address'),
            )
            personal_info_list.append(contact_info)

    return personal_info_list

import concurrent.futures

def process_email(email):
    prompt = create_prompt(email)
    output = llm.predict(prompt)
    input_list = output.split('Person')
    result = parse_contact_info(input_list)
    found_contacts = pd.DataFrame([i.__dict__ for i in result])
    return found_contacts

st.markdown("""
<style>
    .find-contacts-button {
        display: flex;
        padding: 0.375rem 1rem;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        border-radius: 0.25rem;
        background: #0050C3;
        color: #fff; /* Text color for the button (white) */
        text-align: center; /* Center the text in the button */
        font-size: 16px; /* Font size for the button text */
        font-weight: bold; /* Bold font weight for the button text */
        /* Add the specified box shadow */
        box-shadow: 0px 3px 1px -2px rgba(0, 0, 0, 0.20), 0px 2px 2px 0px rgba(0, 0, 0, 0.14), 0px 1px 5px 0px rgba(0, 0, 0, 0.12);
    }

    /* Contact cards styling */
    .contact-card-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        box-sizing: border-box;
        padding: 20px;
        font-family: 'Arial', sans-serif;
        max-width: 800px; /* Limit container width for better readability */
        margin: 0 auto; /* Center the container */
    }

    .contact-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px; /* Add space between rows */
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    .contact-card h2 {
        font-size: 20px;
        font-style: normal;
        font-weight: 500;
        line-height: 160%;
        letter-spacing: 0.15px;
        color: #000;
        font-feature-settings: 'clig' off, 'liga' off;
    }

    .contact-card p.job-title {
        font-size: 14px;
        font-style: normal;
        font-weight: 400;
        line-height: 143%;
        letter-spacing: 0.17px;
        color: #4B4B4B;
        font-feature-settings: 'clig' off, 'liga' off;
    }

    .contact-card p.email-phone {
        font-size: 14px;
        font-style: normal;
        font-weight: 400;
        line-height: 150%;
        letter-spacing: 0.15px;
        color: #000;
        font-feature-settings: 'clig' off, 'liga' off;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Header styling */
    .header {
        display: flex;
        align-items: center;
        gap: 0.1875rem;
        align-self: stretch;
        background: #3C3C3C;
        /* Add the specified box shadow */
        box-shadow: 0px 1px 3px 0px rgba(0, 0, 0, 0.12), 0px 1px 1px 0px rgba(0, 0, 0, 0.14), 0px 2px 1px -1px rgba(0, 0, 0, 0.20);
    }
    /* Logo styling */
    .header-logo {
        width: 50px; /* Adjust the width of the logo */
        height: 50px; /* Adjust the height of the logo */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Text Box styling */
    .text-box {
        display: flex;
        width: 87.875rem;
        height: 16.5rem;
        justify-content: center;
        align-items: center;
        border-radius: 0.5rem;
        background: #FFF;
        /* Add any additional styles you want for the text box */
        /* For example, you can add padding or a box shadow */
        padding: 1rem;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

logo_svg = open("ContactEase icon.svg", "r").read()

def main():
    st.markdown("""
    <div class="header">
        <!-- SVG logo -->
        <div class="header-logo">
            {}
        </div>
        <!-- Add any other elements you want in the header, such as a title or description -->
        <h1 style="color: #fff;">Contact Finder</h1>
    </div>
    """.format(logo_svg), unsafe_allow_html=True)

    st.markdown("")

    with st.container():
        conversation_input = st.text_area("Paste your message below", "", height=500)

    cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # Split the window into three columns of equal width
    
    # Add clear button in the first column
    if cols[0].button("Clear"):
        conversation_input = ""  # This sets the conversation_input to an empty string
        
    # Leave the second column empty
    cols[1].write("")

    # Add the extraction button to the third column
    if cols[11].button("Find Contacts"):
            if conversation_input:

                emails = ['From:' + i for i in conversation_input.split('From:') if len(i) > 10]

                main = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_email = {executor.submit(process_email, email): email for email in emails}
                    from tqdm import tqdm
                    for future in tqdm(concurrent.futures.as_completed(future_to_email)):
                        email = future_to_email[future]
                        try:
                            data = future.result()
                        except Exception as exc:
                            print('%r generated an exception: %s' % (email, exc))
                        else:
                            print('%r extracted successfully' % (email))
                            main.append(data)

                # Custom CSS to reduce the margin between "Found Contacts" heading and the contact cards
                st.markdown("""
                <style>
                    /* Reduce the margin between Found Contacts heading and the contact cards */
                    .found-contacts-heading {
                        margin-bottom: 0.5rem;
                    }
                </style>
                """, unsafe_allow_html=True)

                st.markdown("""<p class="found-contacts-heading" style='color: #404040; font-family: Roboto; font-size: 1.25rem; font-style: normal; font-weight: 400; line-height: normal;'>Found Contacts</p>""", unsafe_allow_html=True)
                
                output = to_lowercase(pd.concat(main, ignore_index=True).replace('N/A', None).replace('n/a', None).replace('', None).drop_duplicates())
                
                deduplicated_df = get_unique_contacts_with_most_info(output)

                # Calculate the number of NaNs in each row
                nan_counts = deduplicated_df.isna().sum(axis=1)

                # Sort the DataFrame based on the least amount of NaNs
                contacts = deduplicated_df.iloc[nan_counts.argsort()].to_dict('records')

                # The style and start of the contact card container
                st.markdown("""
                <style>
                    .contact-card-container {
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 20px;
                        box-sizing: border-box;
                        padding: 20px;
                        font-family: 'Arial', sans-serif;
                        max-width: 800px; /* Limit container width for better readability */
                        margin: 0 auto; /* Center the container */
                    }
                    .contact-card {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 10px;
                        margin-bottom: 20px; /* Add space between rows */
                        font-size: 14px;
                        line-height: 1.6;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                        min-height: 280px; /* Set the minimum height */
                    }
                    .contact-card h2 {
                        font-size: 18px;
                        margin-bottom: 5px;
                    }
                    .contact-card p {
                        margin: 0;
                    }
                </style>
                <div class="contact-card-container">
                """, unsafe_allow_html=True)

                index = 0
                cols = st.columns(3)
 
                for contact in contacts:
                    with cols[index % 3]:
                        first_name = contact['first_name'].capitalize()

                        last_name = contact['last_name'].capitalize()

                        job_title = contact['job_title'] if pd.notnull(contact['job_title']) else ""

                        if job_title != "":
                            job_title = job_title.title()                        

                        company_name = contact['company_name'] if pd.notnull(contact['company_name']) else ""

                        if company_name != "":
                            company_name = company_name.title()

                        email = contact['email'] if pd.notnull(contact['email']) else ""

                        mobile_number = contact['mobile_number'] if pd.notnull(contact['mobile_number']) else ""

                        office_number = contact['office_number'] if pd.notnull(contact['office_number']) else ""

                        business_website = contact['business_website'] if pd.notnull(contact['business_website']) else ""

                        address = contact['address'] if pd.notnull(contact['address']) else ""

                        if address != "":
                            address = address.title()

                        st.markdown(f"""
                            <div class="contact-card">
                                <h2><strong>{first_name} {last_name}</strong></h2>
                                <p class="job-title"><small style="color: grey;">{job_title}</small></p>
                                <p class="company-name">{company_name}</p>
                                <p class="email">{email}</p>
                                <p class="mobile-number">{mobile_number}</p>
                                <p class="office-number">{office_number}</p>
                                <p class="business-website">{business_website}</p>
                                <p class="address">{address}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        index += 1

                # End of the contact card container
                st.markdown("""
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a conversation text.")

if __name__ == "__main__":
    main()