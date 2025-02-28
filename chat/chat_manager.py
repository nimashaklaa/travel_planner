import spacy
from openai import OpenAI

from agents.context_agent.context_agent import generate_constraint_set
from agents.planner_agent.planner_fun import TravelPlanner
from agents.tool_agent.feedback_agent import FeedbackAgent, generate_updated_plan
from agents.tool_agent.tool_agent import generate_one_plan
from db.db_manager import save_profile_to_db
from nlp.nlp_resources import get_nlp_instance
from nlp.topic_categorizer import categorize_topics_with_llm
from utils.environment_management import load_api_key

nlp = get_nlp_instance()

def get_response(prompt):
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    response_content = completion.choices[0].message.content
    return response_content


def handle_chat(user_query: str):
    """Process user input and generate a response."""
    if not user_query.strip():
        return "Error: Empty query received."

    # Generate travel plan based on input
    plan = generate_one_plan(user_query)

    # Return the AI-generated plan
    return plan

# def handle_chat(chat_history_path):
#     print('trying to fetch', chat_history_path)
#     chat_history_path = 'D:/semester 7/FYP/chat_history.txt'
#     with open(chat_history_path, "a") as chat_file:
#         print("Chatbot is ready to talk! Type 'quit' to exit.")
#         while True:
#             user_message = input("You: ")
#             if user_message.lower() == 'quit':
#                 break
#
#             plan, scratchpad, action_log = generate_one_plan( user_message)
#             print("\n📍 Initial Travel Plan:\n", plan)
#
#             while True:
#                 user_feedback = input("\nDo you want to modify anything? Describe your changes or type 'no': ")
#                 if user_feedback.lower() == 'no':
#                     print("\n✅ Keeping the final plan.")
#                     final_plan = plan
#                     print(final_plan)
#                     break
#                 else:
#                     updated_plan, scratchpad, action_log = generate_updated_plan(user_feedback, plan)
#                     plan = updated_plan
#                     print(plan)
#                     print("\n✅ Travel Plan Updated!\n")



def extract_user_messages(chat_content):
    """ Extracts only the user's messages from the chat history. """
    user_messages = []
    for line in chat_content.split("\n"):
        if line.startswith("You:"):
            user_messages.append(line.split(":", 1)[1].strip())
    return " ".join(user_messages)

def create_user_profile(chat_history_path):
    """ Generates a user_profile from chat history by categorizing topics. """
    with open(chat_history_path, "r") as file:
        chat_content = file.read()

    user_messages = extract_user_messages(chat_content)
    print(user_messages)
    doc = nlp(user_messages)
    profile = categorize_topics_with_llm(doc)

    return profile