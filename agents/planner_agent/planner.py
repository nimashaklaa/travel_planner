# Import necessary libraries
import os
from datetime import datetime
from pandas import DataFrame

from agents.context_agent.context_agent import generate_constraint_set
# Import custom or external modules
from tools.accommodations.apis import Accommodations
from tools.attractions.apis import Attractions
from tools.restaurants.apis import Restaurants
from tools.flights.apis import Flights
# from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.cities.apis import Cities
# from tools.planner.apis import Planner
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class Notebook:
    def __init__(self) -> None:
        self.data = []

    def write(self, input_data: DataFrame, short_description: str):
        self.data.append({"Short Description": short_description, "Content": input_data})
        return f"The information has been recorded in Notebook, and its index is {len(self.data) - 1}."

    def update(self, input_data: DataFrame, index: int, short_decription: str):
        self.data[index]["Content"] = input_data
        self.data[index]["Short Description"] = short_decription

        return f"The information has been updated in Notebook."

    def list(self):
        results = []
        for idx, unit in enumerate(self.data):
            results.append({"index": idx, "Short Description": unit['Short Description']})

        return results

    def list_all(self):
        results = []
        for idx, unit in enumerate(self.data):
            if type(unit['Content']) == DataFrame:
                results.append({"index": idx, "Short Description": unit['Short Description'],
                                "Content": unit['Content'].to_string(index=False)})
            else:
                results.append(
                    {"index": idx, "Short Description": unit['Short Description'], "Content": unit['Content']})

        return results

    def read(self, index):
        return self.data[index]

    def reset(self):
        self.data = []

ZEROSHOT_REACT_INSTRUCTION = """Collect information for a query plan using interleaving 'Thought', 'Action', and 'Observation' steps. Ensure you gather valid information related to transportation, dining, attractions, and accommodation. All information should be written in Notebook, which will then be input into the Planner tool. Note that the nested use of tools is prohibited. 'Thought' can reason about the current situation, and 'Action' can have 8 different types:
(1) FlightSearch[Departure City, Destination City, Date]:
Description: A flight information retrieval tool.
Parameters:
Departure City: The city you'll be flying out from.
Destination City: The city you aim to reach.
Date: The date of your travel in YYYY-MM-DD format.
Example: FlightSearch[New York, London, 2022-10-01] would fetch flights from New York to London on October 1, 2022.

(2) GoogleDistanceMatrix[Origin, Destination, Mode]:
Description: Estimate the distance, time and cost between two cities.
Parameters:
Origin: The departure city of your journey.
Destination: The destination city of your journey.
Mode: The method of transportation. Choices include 'self-driving' and 'taxi'.
Example: GoogleDistanceMatrix[Paris, Lyon, self-driving] would provide driving distance, time and cost between Paris and Lyon.

(3) AccommodationSearch[City]:
Description: Discover accommodations in your desired city.
Parameter: City - The name of the city where you're seeking accommodation.
Example: AccommodationSearch[Rome] would present a list of hotel rooms in Rome.

(4) RestaurantSearch[City]:
Description: Explore dining options in a city of your choice.
Parameter: City – The name of the city where you're seeking restaurants.
Example: RestaurantSearch[Tokyo] would show a curated list of restaurants in Tokyo.

(5) AttractionSearch[City]:
Description: Find attractions in a city of your choice.
Parameter: City – The name of the city where you're seeking attractions.
Example: AttractionSearch[London] would return attractions in London.

(6) CitySearch[State]
Description: Find cities in a state of your choice.
Parameter: State – The name of the state where you're seeking cities.
Example: CitySearch[California] would return cities in California.

(7) NotebookWrite[Short Description]
Description: Writes a new data entry into the Notebook tool with a short description. This tool should be used immediately after FlightSearch, AccommodationSearch, AttractionSearch, RestaurantSearch or GoogleDistanceMatrix. Only the data stored in Notebook can be seen by Planner. So you should write all the information you need into Notebook.
Parameters: Short Description - A brief description or label for the stored data. You don't need to write all the information in the description. The data you've searched for will be automatically stored in the Notebook.
Example: NotebookWrite[Flights from Rome to Paris in 2022-02-01] would store the informatrion of flights from Rome to Paris in 2022-02-01 in the Notebook.

(8) Planner[Query]
Description: A smart planning tool that crafts detailed plans based on user input and the information stroed in Notebook.
Parameters: 
Query: The query from user.
Example: Planner[Give me a 3-day trip plan from Seattle to New York] would return a detailed 3-day trip plan.
You should use as many as possible steps to collect engough information to input to the Planner tool. 

Each action only calls one function once. Do not add any description in the action.

Query: {USER_QUERY}"""

COT_PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and hotel names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. Attraction visits and meals are expected to be diverse. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). 

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****
Given information: {text}
Query: {query}
Travel Plan: Let's think step by step. First, """

# user_message = "I want to plan a trip to New Zealand for 4 days with my parents"
# USER_QUERY= generate_constraint_set(user_message)

USER_QUERY = """

Destination: Pensacola, Florida, United States

Duration: 1 days 


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

notebook_instance = Notebook()
Flights = Flights()
#GoogleDistanceMatrix()
Accommodations = Accommodations()
Attractions = Attractions()
Restaurants = Restaurants()
Cities = Cities()

def FlightSearch(departure_city, destination_city, date):
    """Flight information retrieval tool."""
    return Flights.run(departure_city, destination_city, date)

def GoogleDistanceMatrix(origin, destination, mode):
    """Estimate the distance, time and cost between two cities."""
    #return GoogleDistanceMatrix.run(origin, destination, mode)
    return "GoogleDistanceMatrix used"

def AccommodationSearch(city):
    """Discover accommodations in your desired city."""
    return Accommodations.run(city)

def RestaurantSearch(city):
    """Explore dining options in a city of your choice."""
    return Restaurants.run(city)

def AttractionSearch(city):
    """Find attractions in a city of your choice."""
    return Attractions.run(city)

def CitySearch(state):
    """Find cities in a state of your choice."""
    return Cities.run(state)

def NotebookWrite(input_data, short_description):
    """Writes a new data entry into the Notebook tool with a short description.This tool should be used immediately after FlightSearch, AccommodationSearch, AttractionSearch, RestaurantSearch or GoogleDistanceMatrix. Only the data stored in Notebook can be seen by Planner. So you should write all the information you need into Notebook including the Dataframes returned by the other tools."""
    return notebook_instance.write(input_data,short_description)

def Planner(query):
    """A smart planning tool that crafts detailed plans based on user input and the information stored in Notebook."""
    text = notebook_instance.list_all()
    return COT_PLANNER_INSTRUCTION.format(text=text, query=query)

def calculate_sum(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b

def check_wheather(location: str , at_time: datetime | None = None) -> str:
    """"Return the weather at the given location and time."""
    return f"It's sunny today in {location}!"


tools = [FlightSearch, GoogleDistanceMatrix, AccommodationSearch, RestaurantSearch, AttractionSearch, CitySearch,
         NotebookWrite, Planner]

model = ChatOpenAI(model="gpt-4o", api_key=api_key)

system_prompt = ZEROSHOT_REACT_INSTRUCTION

graph = create_react_agent(model, tools=tools, state_modifier=system_prompt)

inputs = {"messages": [("user", f"{USER_QUERY}")]}

for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()

notebook_instance.list_all()