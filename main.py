from agents.planner_agent.planner_fun import TravelPlanner
from agents.user_profile.create_user_profile import get_summery, save_profile_to_txt, save_summary_to_txt
from chat.chat_manager import handle_chat
from db.db_manager import retrieve_profile_from_db


def main():
    planner = TravelPlanner()
    user_query = "Plan a trip from New York to Tokyo."
    planner.run(user_query)
    # chat_history_path = "chat_history.txt"
    # handle_chat(chat_history_path)
    # user_profile = retrieve_profile_from_db()
    # summery = get_summery(user_profile)
    # profile_file_path = "user_profile.txt"
    # save_summary_to_txt(summery, profile_file_path)


if __name__ == "__main__":
    main()