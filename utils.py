from langchain.chat_models import ChatOpenAI
import openai

def get_image(name):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    gender = llm.predict(f"Classify {name} as male or female. If you unsure, refer to female. Your response needs to be male or female.")

    response = openai.Image.create(
    prompt=f"professional business {gender}",
    n=1,
    size="1024x1024"
    )

    return response['data'][0]['url']