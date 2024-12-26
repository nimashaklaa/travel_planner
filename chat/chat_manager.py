import spacy
from openai import OpenAI

from agents.context_agent.context_agent import generate_constraint_set
from agents.planner_agent.planner_fun import TravelPlanner
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


def handle_chat(chat_history_path):
    with open(chat_history_path, "a") as chat_file:
        print("Chatbot is ready to talk! Type 'quit' to exit.")
        while True:
            user_message = input("You: ")
            if user_message.lower() == 'quit':
                break

            if "plan a trip" in user_message.lower():

                constraint_set = generate_constraint_set(user_message)
                print("AI:", constraint_set)
                planner = TravelPlanner()
                planner.run(constraint_set)

            # ai_response = get_response(user_message)
            # print("AI:", ai_response)

            # Write messages to file
            chat_file.write(f"You: {user_message}\n")
            # TODO: chat should output the plan
            chat_file.write(f"AI: {constraint_set}\n")

    # if chat_active:
        # Process chat history to create user_profile
        profile = create_user_profile(chat_history_path)
        save_profile_to_db(profile)
        print("Chat session ended and profile updated.")


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