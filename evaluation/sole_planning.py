import os
import json
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from utils.environment_management import load_api_key

# Load dataset
dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
df = dataset.to_pandas()
df_test = df[["query", "reference_information"]][:20]

# Display sample data
pprint(df_test["reference_information"][0])

# Define planner prompts
PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).

Given information: {text}
Query: {query}
Travel Plan:"""

UPDATED_PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided user profile, information, and query, please give me a detailed and personalized travel plan. The plan should reflect the user's preferences and interests while adhering to the budget and other constraints mentioned in the query. Include specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Ensure that all information in your plan is derived from the provided data and aligns with commonsense.

User Profile: {user_profile}
Given information: {text}
Query: {query}
Travel Plan:"""

# Define LLM and Chain
OPENAI_API_KEY =load_api_key()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=1, model="gpt-4-1106-preview")
planner_chain = LLMChain(llm=llm,
                         prompt=PromptTemplate(input_variables=["text", "query"], template=PLANNER_INSTRUCTION))
updated_planner_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["user_profile", "text", "query"],
                                                                template=UPDATED_PLANNER_INSTRUCTION))

# Generate plans using the initial template
for i in range(len(df_test) - 4):
    input_data = {
        "text": df_test["reference_information"].iloc[i],
        "query": df_test["query"].iloc[i],
    }
    travel_plan = planner_chain.run(input_data)
    print(travel_plan)

# Generate plans using the updated template
for i in range(len(df_test) - 4):
    input_data = {
        "user_profile": "She is a 34-year-old freelance graphic designer who enjoys hiking, yoga, and painting. She is vegan, a spicy-food lover, and loves experimenting with new recipes. She enjoys traveling worldwide and has a particular fondness for natural scenery. She lives in a small apartment in the city with two cats named Pixel and Scribble.",
        "text": df_test["reference_information"].iloc[i],
        "query": df_test["query"].iloc[i],
    }
    travel_plan = updated_planner_chain.run(input_data)
    print(travel_plan)

# Save plans to files
args = {
    "output_dir": './',
    "set_type": "validation",
    "strategy": "direct",
    "model_name": "gpt-4-1106-preview"
}

for i in tqdm(range(len(df_test))):
    input_data = {
        "user_profile": "She is a 34-year-old freelance graphic designer who enjoys hiking, yoga, and painting. She is vegan, a spicy-food lover, and loves experimenting with new recipes. She enjoys traveling worldwide and has a particular fondness for natural scenery. She lives in a small apartment in the city with two cats named Pixel and Scribble.",
        "text": df_test["reference_information"].iloc[i],
        "query": df_test["query"].iloc[i],
    }
    travel_plan = updated_planner_chain.run(input_data)
    print(travel_plan)

    output_path = os.path.join(f'{args["output_dir"]}/{args["set_type"]}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_path = os.path.join(f'{output_path}/generated_plan_{i + 1}.json')
    if not os.path.exists(file_path):
        result = [{}]
    else:
        with open(file_path, 'r') as f:
            result = json.load(f)

    if args["strategy"] in ['react', 'reflexion']:
        result[-1][f'{args["model_name"]}_{args["strategy"]}_sole-planning_results_logs'] = travel_plan
    result[-1][f'{args["model_name"]}_{args["strategy"]}_sole-planning_results'] = travel_plan

    with open(file_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Stored result in {file_path}")
