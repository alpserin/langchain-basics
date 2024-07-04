from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()

# Load our OpenAI api key from the env file
OPENAI_KEY = os.getenv('OPENAI_KEY')

llm = OpenAI(temperature=0.7, api_key=OPENAI_KEY)

def generate_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables = ['cuisine'],
        template = "I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Chain 2: Menu Items
    prompt_template_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template = """Give me a comma seperated text that has menu items for the restaurant named {restaurant_name}."""
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    response = chain({'cuisine': cuisine})

    return response