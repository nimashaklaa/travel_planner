import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Define the column names
column_names = [
    'email', 'event_name', 'duration_minutes', 'created_at',
    'start_time', 'year', 'week_of_year', 'recurring_event',
    'reminder_sent', 'event_priority', 'is_cancelled', 'event_id'
]

# Replace 'your_file.csv' with the actual file name
df = pd.read_csv('sample_data.csv', names=column_names, header=None)

print(df.groupby('email')['event_name'])

def categorize_event_for_profile(event_name):
    """ Use OpenAI to categorize events into predefined user profile categories. """
    # Prompt design: ask GPT to classify the event into the user profile categories
    prompt = f"The following is a calendar event: '{event_name}'. Please categorize it as one of the following categories and response should only return: interest, dislike, food, or hobby. If none of these apply, return 'none'."

    # Make API call to OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the response text (the event category)
    category = completion.choices[0].message.content.strip().lower()

    # Ensure valid categories or return 'none'
    valid_categories = ["interest", "dislike", "food", "hobby"]
    return category if category in valid_categories else "none"


def create_user_profile_with_openai(event_series):
    """ Categorize a user's calendar events into interests, dislikes, foods, and hobbies. """
    profile = {
        "interests": [],
        "dislikes": [],
        "foods": [],
        "hobbies": []
    }

    # Loop through each event and categorize it using OpenAI
    for event in event_series:
        print(event)
        category = categorize_event_for_profile(event)

        # Append the event to the appropriate category in the profile
        if category == "interest":
            profile["interests"].append(event)
        elif category == "dislike":
            profile["dislikes"].append(event)
        elif category == "food":
            profile["foods"].append(event)
        elif category == "hobby":
            profile["hobbies"].append(event)

    return profile

# Apply the profile creation function to the grouped events
user_profiles = df.groupby('email')['event_name'].apply(create_user_profile_with_openai)

# Convert the profiles to a DataFrame for easy viewing
user_profiles_df = pd.DataFrame(list(user_profiles), index=user_profiles.index)

# Display the user profiles
print(user_profiles_df)
