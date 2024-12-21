import os
import pickle
import datetime
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the token.pickle file.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']


def authenticate_google_calendar():
    creds = None
    # The token.pickle stores the user's access and refresh tokens and is created automatically when the authorization flow completes..
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no valid credentials, allow the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds


def get_calendar_events():
    # Authenticate and build the service
    creds = authenticate_google_calendar()
    service = build('calendar', 'v3', credentials=creds)

    # Call the Calendar API to get events from the primary calendar
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=10, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    return events


def clean_description(description):
    # Remove any HTML tags or links (anything starting with 'http' or '<a>')
    description = re.sub(r'<a.*?>.*?</a>', '', description)  # Remove HTML links
    description = re.sub(r'http\S+', '', description)  # Remove URLs
    description = re.sub(r'\s+', ' ', description).strip()  # Remove extra spaces/newlines
    return description


def save_events_to_txt(events, filename='calendar_events.txt'):
    with open(filename, 'w') as file:
        if not events:
            file.write("No upcoming events found.\n")
        else:
            for event in events:
                event_details = f"Event: {event['summary']}\n"

                # Only include description if it exists and is not empty after cleaning
                if 'description' in event:
                    clean_desc = clean_description(event['description'])
                    if clean_desc:  # Only print description if it's not empty
                        event_details += f"Description: {clean_desc}\n"

                event_details += "\n"  # Add a blank line between events
                file.write(event_details)

    print(f"Events saved to {filename}")


events = get_calendar_events()
save_events_to_txt(events)
