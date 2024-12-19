from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv("DATABASE_URL")
client = MongoClient(database_url)
db = client.mydatabase
collections = {
    "interests": db.interests,
    "dislikes": db.dislikes,
    "foods": db.foods,
    "hobbies": db.hobbies
}

collection_interests = db.interests
collection_dislikes = db.dislikes
collection_foods = db.foods
collection_hobbies = db.hobbies

def save_profile_to_db(profile):
    for category, items in profile.items():
        collection = collections[category]
        for item in items:
            collection.insert_one({"event_name": item})
    print("User profile saved to MongoDB.")

def retrieve_profile_from_db():
    """Retrieve user_profile data from the MongoDB database."""
    profile = {}

    # Retrieve interests
    interests = list(collection_interests.find({}))
    profile['interests'] = [interest['event_name'] for interest in interests]

    # Retrieve dislikes
    dislikes = list(collection_dislikes.find({}))
    profile['dislikes'] = [dislike['event_name'] for dislike in dislikes]

    # Retrieve foods
    foods = list(collection_foods.find({}))
    profile['foods'] = [food['event_name'] for food in foods]

    # Retrieve hobbies
    hobbies = list(collection_hobbies.find({}))
    profile['hobbies'] = [hobby['event_name'] for hobby in hobbies]

    return profile
