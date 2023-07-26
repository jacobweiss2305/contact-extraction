import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import utils

load_dotenv()

def main():
    st.title("AI Contact Extraction App")
    st.write("Enter the conversation text:")
    conversation_input = st.text_area("Conversation", "")

    if st.button("Extract Contacts"):
            if conversation_input:
                # Load conversation from user input
                conversation = conversation_input
                doc = Document(page_content=conversation)
                split_docs = RecursiveCharacterTextSplitter().split_documents([doc])

                # Define the 'schema' object (same as in your original script)
                schema = Object(
                    id="personal_info",
                    description="Personal information about a given person.",
                    attributes=[
                        Text(
                            id="first_name",
                            description="The first name of the person",
                            examples=[("John Smith went to the store", "John")],
                        ),
                        Text(
                            id="last_name",
                            description="The last name of the person",
                            examples=[("John Smith went to the store", "Smith")],
                        ),
                        Text(
                            id="job_title",
                            description="The job title of the person",
                            examples=[("John Smith is a sales associate at a local store", "sales associate")],
                        ),
                        Text(
                            id="company_name",
                            description="The company name the person works for",
                            examples=[("John Smith is a sales associate at a walmart", "walmart")],
                        ),
                        Text(
                            id="phone_number",
                            description="The phone number of the person",
                            examples=[("John Smith is a sales associate and his phone number is 719-239-0231", "719-239-0231")],
                        ),
                        Text(
                            id="email",
                            description="The email of the person",
                            examples=[("John Smith is a sales associate and his email is john.smith@email.com", "john.smith@email.com")],
                        ),
                        Text(
                            id="address",
                            description="The address of the company the person works for",
                            examples=[("John Smith works at Data Axle 123 located at Main St, New York, NY 10001", "john.smith@email.com")],
                        )        
                    ],
                    examples=[
                        (
                            """
                            John Smith
                            Senior Sales & Marketing Director
                            
                            719-239-0231
                            john.smith@email.com

                            Data Axle
                            123 Main St, New York, NY 10001
                            
                            Jane Doe
                            Sales Executive
                            
                            719-239-9999
                            jane.doe@email.com

                            KPMG
                            123 Main St, San Franciso, CA 90909
                            """,
                            [
                                {
                                    "first_name": "John", 
                                    "last_name": "Smith", 
                                    "job_title": "Senior Sales & Marketing Director",
                                    "company_name": "Data Axle",
                                    "phone_number": "719-239-0231",
                                    "email": "john.smith@email.com",
                                    "address": "123 Main St, New York, NY 10001"
                                },
                                {
                                    "first_name": "Jane", 
                                    "last_name": "Doe", 
                                    "job_title": "Sales Executive",
                                    "company_name": "KPMG",
                                    "phone_number": "719-239-9999",
                                    "email": "jane.doe@email.com",
                                    "address": "123 Main St, San Franciso, CA 90909"                    
                                },
                            ],
                        )
                    ],
                    many=True,
                )

                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0,
                )

                chain = create_extraction_chain(
                    llm,
                    schema,
                    encoder_or_encoder_class="csv",
                    input_formatter="triple_quotes",
                )

                # Create an event loop and run the asynchronous part in a separate thread
                loop = asyncio.new_event_loop()

                asyncio.set_event_loop(loop)

                document_extraction_results = loop.run_until_complete(
                    utils.extract_contacts_from_documents(chain, split_docs)
                )

                df = utils.generate_dataframe(document_extraction_results)

                st.write("Extracted Contacts:")

                st.write(df)
            else:
                st.warning("Please enter a conversation text.")

if __name__ == "__main__":
    main()
