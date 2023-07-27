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

import concurrent.futures



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
    st.title("AI Contact Extraction App")
    st.write("Enter the conversation text:")
    conversation_input = st.text_area("Conversation", "")

    if st.button("Extract Contacts"):
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

                st.write("Extraction Results:")

                output = to_lowercase(pd.concat(main, ignore_index=True).drop_duplicates().replace('', None))
                
                deduplicated_df = get_unique_contacts_with_most_info(output)
                
                st.dataframe(deduplicated_df)
            else:
                st.warning("Please enter a conversation text.")

if __name__ == "__main__":
    main()
