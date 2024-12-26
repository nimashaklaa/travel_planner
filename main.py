
from chat.chat_manager import handle_chat




def main():
    chat_history_path = "chat_history.txt"
    handle_chat(chat_history_path)


# TODO: improve the NLP classifier
if __name__ == "__main__":
    main()