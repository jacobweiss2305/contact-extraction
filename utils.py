import pandas as pd

from langchain.callbacks import get_openai_callback
from kor import extract_from_documents

# __import__('pysqlite3')
# import sys
# import os
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': os.path.join('/workspaces/contact-extraction', 'db.sqlite3'),
#     }
# }

def load_conversation(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        conversation = f.read()

    return conversation

def generate_dataframe(json_data):
    # Prepare an empty list to store all restaurant data
    data = []

    for record in json_data:
        restaurant_list = record.get('data', {}).get('personal_info', [])
        for restaurant in restaurant_list:
            # Get details for each restaurant and append it to data
            data.append([
                restaurant.get('first_name', ''),
                restaurant.get('last_name', ''),
                restaurant.get('job_title', ''),
                restaurant.get('mobile_number', ''),
                restaurant.get('desk_number', ''),
                restaurant.get('email', ''),
                restaurant.get('company_name', ''),
                restaurant.get('business_url', ''),  
                restaurant.get('address', ''),

            ])

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=[
        'first_name', 'last_name', 'job_title', 
        'mobile_number', 'desk_number', 'email', 
        'company_name', 'business_url', 'address',
        ])

    return df

async def extract_contacts_from_documents(chain, split_docs):

    with get_openai_callback() as cb:
        return await extract_from_documents(
            chain, split_docs, max_concurrency=5, use_uid=False, return_exceptions=True
        )