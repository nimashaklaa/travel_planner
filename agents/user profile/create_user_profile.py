import os

from dotenv import load_dotenv
from openai import OpenAI

from chat.chat_manager import extract_user_messages
from db.db_manager import retrieve_profile_from_db
from nlp.nlp_resources import get_nlp_instance
from nlp.topic_categorizer import categorize_topics_with_llm

nlp = get_nlp_instance()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def create_user_profile(chat_history_path):
    """ Generates a user profile from chat history by categorizing topics. """
    with open(chat_history_path, "r") as file:
        chat_content = file.read()

    user_messages = extract_user_messages(chat_content)
    print(user_messages)
    doc = nlp(user_messages)
    profile = categorize_topics_with_llm(doc)

    return profile

def save_profile_to_txt(profile, filename):
    """Save the user profile to a text file."""
    with open(filename, 'w') as file:
        for category, items in profile.items():
            file.write(f"{category.capitalize()}:\n")
            if items:
                file.write(", ".join(items) + "\n")
            else:
                file.write("None\n")
            file.write("\n")  # Add a newline for better readability


# Save the user profile to a text file
profile_file_path = "user_profile.txt"
user_profile = retrieve_profile_from_db()

def get_summery(prompt):
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "user", "content":  f"Please provide a concise summary of the following user profile information: "
                        f"{prompt}. "
                        f"The summary should highlight the user's preferences, interests, dislikes, favorite foods, "
                        f"and any hobbies they may have."}
        ]
        )
    response_content = completion.choices[0].message.content
    return response_content

summery =  get_summery(user_profile)
print(summery)

#ToDO: need to print the summery to a text