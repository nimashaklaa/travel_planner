from agents.tool_agent.prompts import zeroshot_react_agent_prompt
from agents.tool_agent.tool_agent import ReactAgent
from chat.chat_manager import handle_chat

user_query=[
    # "Please create a travel plan for me where I'll be departing from Washington and heading to Myrtle Beach for a 3-day trip from March 13th to March 15th, 2022. Can you help me keep this journey within a budget of $1,400?",
    # "Please draw up a 3-day travel itinerary for one person, beginning in Oakland and heading to Tucson from March 15th to March 17th, 2022, with a budget of $1,400.",
    # "Can you help me with a travel plan departing from Buffalo to Atlanta for a duration of 3 days, specifically from March 2nd to March 4th, 2022? I plan to travel alone and my planned budget for the trip is around $1,100.",
    # "Could you arrange a 3-day solo trip for me starting from Ontario and heading to Honolulu spanning from March 4th to March 6th, 2022, with a total budget of $3,200?",
    # "Please assist me in devising a travel plan that departs from West Palm Beach and heads to Atlanta, lasting 3 days from March 13th, 2022 to March 15th, 2022. It should accommodate 1 person and adhere to a budget of $900.",
    # "Please assist in crafting a travel plan for a solo traveller, journeying from Detroit to San Diego for 3 days, from March 5th to March 7th, 2022. The travel plan should accommodate a total budget of $3,000.",
    # "Please create a travel plan for a 3-day trip from Missoula to Dallas scheduled from March 23rd to March 25th, 2022. The budget for this trip is set at $1,900.",
    # "Could you arrange a 3-day travel from Boston to San Juan, Puerto Rico, for one person between March 28th and March 30th, 2022? The budget for this trip is set to $1,400.",
    # "Can you help me plan a trip that begins in Sarasota and ends in Philadelphia? The trip should span over 3 days, from March 2nd to March 4th, 2022, and adhere to a budget of $2,100.",
    # "Could you help me create a travel plan starting from Minneapolis to St. Louis, spanning 3 days from March 15th to March 17th, 2022? The budget is set at $1,000.",
    # "Please create a travel plan departing from Minneapolis and heading to Seattle for 3 days, from March 29th to March 31st, 2022, with a budget of $1,800.",
    # "Please devise a travel plan that starts from St. Petersburg and heads to Appleton, taking place across 3 days from March 19th to March 21st, 2022. This itinerary is for an individual, with a budget allocated at $1,200.",
    # "Can you assist with a travel plan for one person departing from Pittsburgh to Baltimore for 3 days, from March 4th to March 6th, 2022, with a maximum budget of $1,200?",
    # "Could you arrange a travel plan for me starting from Denver and going to Appleton for 3 days, specifically from March 4th to March 6th, 2022? I'm traveling alone and I have a budget of $1,800 for this trip.",
    # "Please arrange a 3-day trip for me departing from St. Louis and visiting Las Vegas from March 29th to March 31st, 2022. My budget for this journey is $1,300.",
    # "Could you help me organize a 3-day journey from Spokane to San Francisco from March 3rd to March 5th, 2022? This trip is for 1 person with a budget of $1,200.",
    # "Can you help draft a 3-day travel plan, starting on March 1st, 2022 and ending on March 3rd, 2022, for one person departing from St. Louis and heading to Washington with a budget of $1,500?",
    # "Could you design a 3-day travel itinerary from Denver to Palm Springs for 1 person? The travel should span from March 27th to March 29th, 2022. The travel budget is set at $2,200. No specific local constraints are given.",
    # "Could you assist in creating a travel plan for one person departing from Seattle and visiting San Francisco for 3 days, from March 21st to March 23rd, 2022? The new budget is $900.",
    # "Could you assist with a 3-day travel itinerary starting from Providence to Orlando, with a visit planned to only one city? The travel dates are from March 24th to March 26th, 2022, and the budget for the trip should not exceed $1,800.",
    "Could you help create a travel itinerary for a solo trip departing from St. Louis and covering 2 cities in Florida over the course of 5 days, from March 15th to March 19th, 2022? The travel budget is set at $2,900.",
    # "Can you help craft a 5-day travel plan that starts in Colorado Springs and takes in 2 cities in Illinois from March 5th to March 9th, 2022? Single traveler with an overall budget of $1,900.",
    # "Could you create a 5-day travel plan for one person departing from Little Rock and visiting 2 cities in Texas from March 14th to March 18th, 2022? The budget for this trip is set at $3,900.",
    # "Could you arrange a 5-day trip for one person, starting from Latrobe and covering two cities in South Carolina from the dates of March 2nd to March 6th, 2022? My budget is set at $4,200.",
    # "Can you create a 5-day travel itinerary for a solo trip starting from Jacksonville and visiting 2 cities in Michigan? The trip should be from March 25th to March 29th, 2022, and I have a budget of $4,600.",
    "Could you construct a 5-day travel itinerary for a solo traveler starting in Orlando and visiting 2 cities in Illinois, spanning the dates from March 2nd to March 6th, 2022? The budget for the trip is set to $2,700.",
    # "Can you create a 5-day travel plan for me that begins in Billings and includes visits to 2 cities in Minnesota? The journey should take place from March 6th to March 10th, 2022, with a budget of $4,000.",
    # "Could you create a 5-day travel plan for me, beginning in Washington, visiting 2 cities in Virginia from March 15th to March 19th, 2022? I have a budget of $2,200 for this trip.",
    # "Can you assist in crafting a travel schedule departing from Richmond and traveling to 2 cities in Tennessee? The journey will last 5 days, starting on March 5th and concluding on March 9th, 2022. The travel budget is set at $2,600.",
    # "Can you help me devise a travel plan that begins in Key West and covers 2 cities in Indiana? The travel dates are from March 10th to March 14th, 2022, and the budget for the trip is $2,000.",
    "Can you help me devise a 5-day travel plan starting from Cedar Rapids and covering 2 cities in Colorado from March 23rd to March 27th, 2022? This journey is for one person with a budget of $4,300.",
    "Can you assist in creating a 5-day long itinerary? I am planning to leave Omaha on March 2nd, 2022, visit 2 different cities in Washington, and return by March 6th, 2022. I will be traveling alone with a budget set to $5,000.",
    # "Could you aid in curating a 5-day travel plan for one person beginning in Denver and planning to visit 2 cities in Washington from March 23rd to March 27th, 2022? The budget for this trip is now set at $4,200.",
    # "Could you put together a 5-day travel plan starting in Charlotte and visiting 2 cities in New Jersey? The dates of travel are from March 6th to March 10th, 2022, and I have a budget of $4,200.",
    # "Can you curate a 5-day travel itinerary for one person starting in Gainesville and visiting 2 cities in North Carolina, from March 23rd to March 27th, 2022? The budget for this plan is set at $2,900.",
    # "Could you please create a 5-day travel plan for me, starting from Los Angeles and visiting 2 cities in Colorado between March 13th and March 17th, 2022? My budget for the trip is $4,700.",
    # "Could you please create a 5-day travel itinerary for one person, starting in Albuquerque and visiting 2 cities in Texas from March 25th to March 29th, 2022? The travel plan should work within a budget of $2,100.",
    # "Could you create a travel plan for a solo traveler starting from Birmingham and visiting 2 cities in Florida, for a duration of 5 days, from March 26th to March 30th, 2022? The allotted budget for this trip is $3,500.",
    # "Could you organize a 5-day travel plan leaving from Killeen and visiting 2 cities in Texas from March 3rd to March 7th, 2022, for one person? The budget for this trip is set at $3,500.",
    # "Can you create a 5-day travel plan for me starting in Sun Valley and visiting 2 cities in California from March 22nd to March 26th, 2022? My budget for this trip is $2,600.",
    # "Can you help construct a travel plan that begins in Philadelphia and includes visits to 3 different cities in Virginia? The trip duration is for 7 days, from March 15th to March 21st, 2022, with a total budget of $1,800.",
    # "Could you construct a week-long travel plan for me, beginning in Bakersfield and heading to Texas? This journey spans from March 2nd to March 8th, 2022, and I am aiming to explore 3 unique cities. I have set aside a budget of $6,100 for this trip.",
    "Could you help me arrange a 7-day solo travel itinerary from Kona to California with a budget of $5,800, intending to visit 3 distinct cities in California from March 7th to March 13th, 2022?",
    # "Can you assist in devising a 7-day trip for one person, commencing in Medford and involving visits to 3 distinct cities in Colorado from March 23rd to March 29th, 2022? The budget for this trip is set at $2,400.",
    # "Could you help me design a one-week travel itinerary departing from Devils Lake and heading to Colorado, covering a total of 3 cities? The travel dates are set between March 22nd and March 28th, 2022. This trip is for a single person with a budget of $3,500.",
    # "Please help me craft a 7-day travel plan for one person departing from Charlotte in North Carolina, visiting 3 cities in Pennsylvania. The travel dates are from March 6th to March 12th, 2022. The allocated budget for this trip is $5,100.",
    "Could you assist me in creating a travel plan from Palm Springs to Texas that spans 7 days, visiting 3 cities from March 13th to March 19th, 2022? I have set aside a budget of $8,100 for this trip.",
    # "I require a travel itinerary for a seven-day trip beginning on March 2nd and ending on March 8th, 2022. The trip will begin in Philadelphia and involve visiting 3 cities in Virginia. The available budget for the trip is $2,900.",
    # "Could you formulate a travel itinerary for me? I'm planning a solo trip from Dallas to Nebraska for 7 days, from March 7th to March 13th, 2022. During this trip, I wish to visit 3 different cities within Nebraska. My budget for this trip is $5,600.",
    # "Can you devise a week-long travel plan for a solo traveler? The trip takes off from Columbus and involves visiting 3 distinct cities in Texas from March 1st to March 7th, 2022. The budget for this venture is set at $4,200.",
    # "Could you assist in creating a week-long travel plan for one person, starting from Indianapolis and venturing through 3 cities in North Carolina from March 7th to March 13th, 2022? The planned budget for the trip is $6,500.",
]


def main():
    chat_history_path = "chat_history.txt"
    handle_chat(chat_history_path,user_query)




# TODO: improve the NLP classifier
if __name__ == "__main__":
    main()