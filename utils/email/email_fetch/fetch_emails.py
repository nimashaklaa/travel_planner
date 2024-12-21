from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path

from nlp.nlp_resources import get_nlp_instance
from nlp.topic_categorizer import categorize_topics_with_llm

# Scopes define the access you're requesting. Read-only access to Gmail inbox:
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

nlp = get_nlp_instance()

def get_credentials():
    creds = None
    # Check if `token.json` file exists (where OAuth token will be stored after authorization)
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no valid credentials available, prompt the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def fetch_messages(service, user_id='me', label_ids=None, max_results=10):
    """Fetches messages from the user's mailbox."""
    results = service.users().messages().list(userId=user_id, labelIds=label_ids, maxResults=max_results).execute()
    messages = results.get('messages', [])
    if not messages:
        print('No messages found.')
        return []
    else:
        detailed_messages = []
        for message in messages:
            msg = service.users().messages().get(userId=user_id, id=message['id']).execute()
            detailed_messages.append(f"{msg['snippet']}")
        return detailed_messages


def fetch_emails():
    """Main function to get credentials and fetch messages."""
    creds = get_credentials()
    # Now, call the Gmail API
    service = build('gmail', 'v1', credentials=creds)

    print("Received Messages:")
    received_messages = fetch_messages(service, max_results=10)
    for message in received_messages:
        print(message)

    print("\nSent Messages:")
    sent_messages = fetch_messages(service, label_ids=['SENT'], max_results=10)
    for message in sent_messages:
        print(message)

    with open('emails_output.txt', 'w') as file:
        for message in received_messages + sent_messages:
            file.write(message + "\n")

def create_user_profile_from_email(email_output_path):
    """ Generates a user_profile from chat history by categorizing topics. """
    with open(email_output_path, "r") as file:
        email_content = file.read()
    doc = nlp(email_content)
    profile = categorize_topics_with_llm(doc)
    return profile

def save_profile(profile):
    """Saves the generated profile to a file or database."""
    with open('profile_output.txt', 'w') as file:
        file.write(str(profile))
    print("Profile saved successfully.")



# fetch_emails()

email_output_path = 'emails_output.txt'

profile = create_user_profile_from_email(email_output_path)

save_profile(profile)

