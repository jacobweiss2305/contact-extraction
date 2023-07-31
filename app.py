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

st.set_page_config(layout="wide")

class ContactInfo(BaseModel):
    first_name: str
    last_name: str
    job_title: str
    company_name: str
    mobile_number: str
    office_number: str
    business_website: str
    email: str
    address: str

class PersonalInfo(BaseModel):
    personal_info: List[ContactInfo]


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def to_lowercase(df):
    return df.applymap(lambda s: s.lower() if type(s) == str else s)

def get_unique_contacts_with_most_info(df):
    # Add a column that counts the number of non-null fields for each contact
    df['info_count'] = df.notnull().sum(axis=1)
    # Group by 'first_name', 'last_name', 'email', and get the row with max 'info_count' for each group
    df = df.loc[df.groupby(['first_name', 'last_name', 'email'])['info_count'].idxmax()]
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

                prompt_msgs = [
                    SystemMessage(
                        content="You are a world class algorithm for extracting information in structured formats."
                    ),
                    HumanMessage(
                        content="Use the given format to extract contact details from the following email (include contacts in Cc:):"
                    ),
                    HumanMessagePromptTemplate.from_template("{input}"),
                    HumanMessage(content="Make sure to answer in the correct format"),
                ]
                prompt = ChatPromptTemplate(messages=prompt_msgs)

                chain = create_structured_output_chain(PersonalInfo, llm, prompt, verbose=True)

                split_emails = split_text_by_from(conversation_input)

                main = []
                # Use ThreadPoolExecutor to parallelize the processing of emails
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit each email processing task to the ThreadPoolExecutor
                    future_to_email = {executor.submit(process_email, chain, email): email for email in split_emails}

                    # Wait for all tasks to complete and get the results
                    for future in concurrent.futures.as_completed(future_to_email):
                        email = future_to_email[future]
                        try:
                            df = future.result()
                        except Exception as exc:
                            df = pd.DataFrame()
                        
                        main.append(df)
                
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
                output = to_lowercase(pd.concat(main, ignore_index=True).drop_duplicates().replace('', None))
                
                deduplicated_df = get_unique_contacts_with_most_info(output)

                contacts = deduplicated_df.to_dict('records')

                output = to_lowercase(pd.concat(main, ignore_index=True).drop_duplicates().replace('', None))

                deduplicated_df = get_unique_contacts_with_most_info(output)

                contacts = deduplicated_df.to_dict('records')
           
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
                        min-height: 170px; /* Set the minimum height */
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
                        email = contact['email'] if pd.notnull(contact['email']) else ""
                        mobile_number = contact['mobile_number'] if pd.notnull(contact['mobile_number']) else ""
                        office_number = contact['office_number'] if pd.notnull(contact['office_number']) else ""

                        st.markdown(f"""
                            <div class="contact-card">
                                <h2><strong>{first_name} {last_name}</strong></h2>
                                <p class="job-title"><small style="color: grey;">{job_title}</small></p>
                                <p class="email-phone">{email}</p>
                                <p class="email-phone">{mobile_number}</p>
                                <p class="email-phone">{office_number}</p>
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