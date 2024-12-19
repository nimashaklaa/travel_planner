import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Function to read user profile from file
def read_user_profile(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print("The specified file was not found.")
        return None

# Read user profile text from a file
user_profile_txt = read_user_profile('user_profile.txt')

# Check if user_profile_txt is not None
if user_profile_txt is None:
    exit()

user_profile_txt_1 = """Sarah Johnson, a 32-year-old marketing manager from Seattle with an annual income of $85,000, enjoys exploring new cultures and cuisines through her frequent work and leisure travels. 
Her travel style blends cultural exploration and relaxation, often leading her to destinations in Europe, Southeast Asia, and notable National Parks. 
Sarah is single and has a cat named Whiskers who requires pet-sitting, she maintains a close-knit family. 
With a budget of around $3,000 for leisure travel, she prioritizes mid-range accommodations and local dining experiences, loves trying local dishes, and remains open to vegetarian options. 
Preferring flights for long distances and road trips for nearby destinations, she utilizes public transportation in cities. Sarah is health-conscious and ensures she has travel insurance while managing a peanut allergy. 
Her travel goals include cultural exposure, relaxation, and adventure, though she faces challenges balancing work responsibilities, securing reliable pet care, and creating itineraries that satisfy her adventurous spirit. 
Preferring to book through online platforms, she frequently seeks inspiration from travel blogs and vlogs."""

original_query_1 = "Can you provide a travel plan for 1 person departing from Kansas City to Pensacola for 3 days, from March 27th to March 29th, 2025, with a budget of $900?"

Generated_Constraint_list_1 = """
Destination: Pensacola, Florida, United States
Duration: 3 days 
Budget: Maximum leisure travel budget: $3,000.
Family and Pet Considerations:  Single, no children.Requires pet-sitting for her cat 
Health and Safety:  Allergic to peanuts; needs dietary accommodations.
Travel Style:   Prefers a mix of guided tours and self-exploration.Seeks cultural exposure, relaxation, and adventure.Mountain climbing.
Accommodations : Prefer eco-friendly accommodations (e.g., eco-lodges or nature-based retreats).
Transportation: Prefers flights for long distances.Enjoys road trips for nearby destinations.Uses public transportation and rideshares in urban areas.
Food Preferences: Enjoys local dishes; open to vegetarian options.
Activities:Interested in outdoor activities (e.g., hiking, kayaking).Prioritizes must-see attractions and cultural experiences.
Other:
"""
user_profile_txt_2 = """Alex Thompson, a 26-year-old software engineer based in Austin, Texas, earns an annual income of $92,000 and has a passion for outdoor activities and exploring lesser-known travel spots. With an adventurous yet budget-conscious travel style, Alex gravitates towards destinations with vibrant outdoor scenes, such as Central America, the Caribbean, and scenic parts of the U.S. Pacific Northwest.
Alex, currently in a long-distance relationship, owns a dog named Luna who needs care while he's away. Close with his family, he often plans trips that allow him to meet with them on the way. With a typical leisure travel budget of around $2,500, Alex prefers affordable accommodations, often choosing hostels, guesthouses, or eco-lodges that provide a balance of comfort and local culture. A food enthusiast, he enjoys sampling street food and regional specialties, especially where vegetarian options are available.
For his travels, Alex tends to opt for budget airlines and rental cars for flexibility, though he’s comfortable using public transit in larger cities. He’s mindful of safety and always ensures he has travel insurance, as he manages a mild shellfish allergy. His travel goals include outdoor adventures like surfing , Mountain climbing, cultural immersion, and personal relaxation, though he sometimes finds it challenging to plan trips around work deadlines, securing pet care, and finding companions for more remote destinations. Alex often books trips through mobile apps, drawing travel ideas from social media, forums, and adventure blogs.
Recently, Alex’s travels have included a week-long adventure in Costa Rica, where he explored lush rainforests, zip-lined through the canopy, and hiked around Arenal Volcano, all while staying in eco-friendly lodges. He embraced the local food scene, sampling street foods, and even tried surfing along the Pacific coast. He also took a long weekend trip to Vancouver, Canada, where he hiked Grouse Mountain and explored the trails of Stanley Park, followed by a day trip to Whistler. Known for its vibrant food scene, Vancouver gave him the chance to try new dishes. Additionally, Alex went on a road trip from Austin to New Mexico, visiting Santa Fe and Taos, where he immersed himself in the region’s unique art installations and scenic desert landscapes. 
Camping along the way, he topped off the trip with a sunrise hot air balloon ride—a highlight of his journey."""

original_query_2 = "Plan a trip to Texas"

Generated_Constraint_list_2 = """
Destination: Texas, United States
Duration: 4 days
Budget:A round $2,500 total, including travel, accommodation, activities, and meals.
Family and Pet Considerations:Has a dog, Luna.
Health and Safety:  Ensure options for travel insurance and access to nearby medical facilities if needed. Avoid restaurants and venues with high risk of shellfish contamination due to Alex’s allergy.
Travel Style:Adventure and outdoor-focused with opportunities for cultural immersion.
Accommodations : Prefer eco-friendly accommodations (e.g., eco-lodges or nature-based retreats).
Transportation: Road trip format is ideal, or budget-friendly flights if needed for specific destinations.Public transit or rental car for flexibility within cities.
Food Preferences:   Local food experiences focusing on street food and regional dishes, with vegetarian options available.
Activities: Outdoor adventures, such as hiking, camping, and nature trails.Explore local culture, including art scenes, historical sites, or unique small towns.
Other:  Draw itinerary ideas from travel blogs, social media, and adventure travel forums for inspiration specific to Texas.
"""
prompt = f'''You are a travel assistant tasked with creating a detailed constraint set for planning a trip based on the user's profile and their trip query. Use the provided user profile text to extract relevant information and define specific constraints that the planner agent should consider while creating the travel itinerary. Ensure constraints are destination-specific, drawing only relevant activities or preferences from the user's profile that suit the location.
Use this format for output:

***** Example 1 *****
User Profile Text:{user_profile_txt_1}
Original Query: {original_query_1}
Generated constraint set: {Generated_Constraint_list_1}
***** Example 1 Ends *****

***** Example 2 *****
User Profile Text:{user_profile_txt_2}
Original Query: {original_query_2}
Generated constraint set: {Generated_Constraint_list_2}
***** Example 2 Ends *****
'''

#
# response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system",
#                  "content": prompt},
#
#                 {"role": "user",
#                  "content": f"user profile text {user_profile_txt} \n\nOriginal Query: {original_query}\n\nPersonalized Query:"}
#             ]
#         )
# print(response.choices[0].message.content)

# Function to generate travel plans based on user profile
def generate_constraint_set(original_query):
    prompt = f'''You are a travel assistant tasked with creating a detailed constraint set for planning a trip based on the user's profile and their trip query. Use the provided user profile text to extract relevant information and define specific constraints that the planner agent should consider while creating the travel itinerary. Ensure constraints are destination-specific, drawing only relevant activities or preferences from the user's profile that suit the location.
    Use this format for output:

    ***** Example 1 *****
    User Profile Text:{user_profile_txt_1}
    Original Query: {original_query_1}
    Generated constraint set: {Generated_Constraint_list_1}
    ***** Example 1 Ends *****

    ***** Example 2 *****
    User Profile Text:{user_profile_txt_2}
    Original Query: {original_query_2}
    Generated constraint set: {Generated_Constraint_list_2}
    ***** Example 2 Ends *****
    '''

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"user profile text {user_profile_txt} \n\nOriginal Query: {original_query}\n\nPersonalized Query:"}
        ]
    )
    return response.choices[0].message.content