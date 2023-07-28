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
    desk_number: str
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

def main():
    st.title("Contact Finder")
    st.markdown("")
    st.markdown("### Paste your message below")

    with st.container():
        conversation_input = st.text_area("", "", height=500)

    cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # Split the window into three columns of equal width
    
    # Add clear button in the first column
    if cols[0].button("Clear"):
        conversation_input = ""  # This sets the conversation_input to an empty string
        
    # Leave the second column empty
    cols[1].write("")

    # Add the extraction button to the third column
    if cols[9].button("Find Contacts"):
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
                
                st.markdown("")                        

                st.markdown("### Found Contacts")

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
                    }
                    .contact-card {
                        border: 2px solid #ccc;
                        border-radius: 5px;
                        padding: 20px;
                        font-size: 0.85em;
                        line-height: 1.6;
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
                        job_title = contact['job_title'] if pd.notnull(contact['job_title']) else "N/A"
                        email = contact['email'] if pd.notnull(contact['email']) else "N/A"
                        mobile_number = contact['mobile_number'] if pd.notnull(contact['mobile_number']) else "N/A"

                        image_url = utils.get_image(f"{first_name} {last_name}")


                        st.markdown(f"""
                            <div class="contact-card">
                                <img src="{image_url}" alt="Profile Picture" width="100">
                                <h2>{first_name} {last_name}</h2>
                                <p><small style="color: grey;">{job_title}</small></p>
                                <p><strong>Email:</strong> {email}</p>
                                <p><strong>Phone:</strong> {mobile_number}</p>
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
