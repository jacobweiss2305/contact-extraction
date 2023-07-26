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
                            id="mobile_number",
                            description="The mobile or direct number of the person",
                            examples=[("John Smith is a sales associate and mobile: 719-239-0231", "719-239-0231")],
                        ),
                        Text(
                            id="desk_number",
                            description="The desk number of the person",
                            examples=[("John Smith is a sales associate and desk phone number: 719-239-0231", "719-239-0231")],
                        ),
                        Text(
                            id="business_website",
                            description="The business website of the company that the person works for",
                            examples=[("John Smith is a sales associate at Data Axle (data-axle.com) and his desk phone number is 719-239-0231", "data-axle.com")],
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
                            Email Chain between Alice and Bob:

                            -------------------------------------------------------

                            **Email 1 - From Alice to Bob**

                            Subject: Planning the Project

                            Hi Bob,

                            I hope this email finds you well. I wanted to discuss the upcoming project with you. Are you available for a quick call tomorrow at 10 AM? Let me know your thoughts.

                            Looking forward to working together!

                            Best regards,
                            Alice Johnson
                            Project Manager
                            ABC Corporation
                            Email: alice.johnson@abc-corp.com
                            Mobile: (555) 111-1111
                            Desk: (555) 123-4567

                            -------------------------------------------------------

                            **Email 2 - From Bob to Alice**

                            Subject: Re: Planning the Project

                            Hi Alice,

                            Thanks for reaching out! A call tomorrow at 10 AM sounds good to me. Let's connect and discuss the project in detail.

                            Best regards,
                            Bob Smith
                            Lead Developer
                            XYZ Solutions
                            Email: bob.smith@xyz-solutions.com
                            Mobile: (555) 222-2222
                            Desk: (555) 987-6543

                            -------------------------------------------------------

                            **Email 3 - From Alice to Bob**

                            Subject: Re: Planning the Project

                            Hi Bob,

                            Great! I've scheduled the call for tomorrow at 10 AM. We'll cover the project scope, timelines, and resource allocation.

                            Looking forward to a productive discussion.

                            Best regards,
                            Alice Johnson
                            Project Manager
                            ABC Corporation
                            Email: alice.johnson@abc-corp.com
                            Mobile: (555) 111-1111
                            Desk: (555) 123-4567

                            -------------------------------------------------------

                            """,
                        [
                            {
                                "first_name": "Alice",
                                "last_name": "Johnson",
                                "job_title": "Project Manager",
                                "company_name": "ABC Corporation",
                                "mobile_number": "(555) 111-1111",
                                "desk_number": "(555) 123-4567",
                                "business_website": "abc-corp.com",
                                "email": "alice.johnson@abc-corp.com",
                                "address": "123 Main St, New York, NY 10001"
                            },
                            {
                                "first_name": "Bob",
                                "last_name": "Smith",
                                "job_title": "Lead Developer",
                                "company_name": "XYZ Solutions",
                                "mobile_number": "(555) 222-2222",
                                "desk_number": "(555) 987-6543",
                                "business_website": "xyz-solutions.com",
                                "email": "bob.smith@xyz-solutions.com",
                                "address": "456 Elm St, San Francisco, CA 90909"
                            }
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

                st.write("Extraction Results:")

                st.write(document_extraction_results)

                df = utils.generate_dataframe(document_extraction_results)

                st.write("Extracted Contacts:")

                st.write(df.drop_duplicates().dropna(how="all"))
            else:
                st.warning("Please enter a conversation text.")

if __name__ == "__main__":
    main()
