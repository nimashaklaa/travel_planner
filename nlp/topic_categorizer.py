import os

from dotenv import load_dotenv
from openai import OpenAI

from nlp.nlp_resources import get_nlp_instance
from nlp.sentiment_analyzer import analyze_sentiment

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

nlp = get_nlp_instance()

# def categorize_topics(doc):
#     """ Categorize extracted topics based on predefined simple keywords and sentiment. """
#     profile = {
#         "interests": [],
#         "dislikes": [],
#         "foods": [],
#         "hobbies": []
#     }
#     food_keywords = ['pizza', 'sea foods', 'soft drinks','sushi', 'pasta', 'burger', 'salad', 'chocolate']
#     hobby_keywords = ['reading', 'jogging', 'gaming', 'cooking', 'photography', 'painting']
#
#     for chunk in doc.noun_chunks:
#         # Adjust to correctly interpret negations and context
#         if "not" in chunk.text.lower() or "n't" in chunk.text.lower():
#             sentiment = -1 * analyze_sentiment(chunk.text)  # Invert sentiment if negation is detected
#         else:
#             sentiment = analyze_sentiment(chunk.text)
#
#         topic = chunk.text.lower().strip()
#
#         # Categorize based on sentiment
#         if sentiment > 0.05:
#             profile["interests"].append(topic)
#         elif sentiment < -0.05:
#             profile["dislikes"].append(topic)
#
#         # Categorize based on keywords
#         if any(food in topic for food in food_keywords):
#             profile["foods"].append(topic)
#         if any(hobby in topic for hobby in hobby_keywords):
#             profile["hobbies"].append(topic)
#
#     return profile

def break_into_sentences_with_llm(prompt):
    """ Use a language model to categorize noun chunks into foods, hobbies, or other categories."""
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": f"Please split the following text into individual sentences:\n\n{prompt}\n\nSentences:"}
      ]
    )
    response_content = completion.choices[0].message.content
    sentences_array = [sentence.split('. ', 1)[-1] for sentence in response_content.strip().split('\n') if sentence]

    return sentences_array


def categorize_with_llm(prompt):
    """ Use a language model to categorize noun chunks into foods, hobbies, or other categories."""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user",
             "content": f"Identify the category of the following topic. It should be classified as either food, hobbies, interests, or dislikes: {prompt}.the response only contains food', 'hobbies', 'interests', 'dislikes', 'none'"}
        ]
    )
    response_content = completion.choices[0].message.content
    valid_categories = {'food', 'hobbies', 'interests', 'dislikes', 'none'}

    # Check if the response is in the valid categories
    if response_content in valid_categories:
        return response_content
    else:
        return 'none'  # Default to 'none' if the response doesn't match any category

def categorize_topics_with_llm(doc):
    """ Categorize extracted topics using LLM categorization and sentiment analysis. """
    profile = {
        "interests": [],
        "dislikes": [],
        "foods": [],
        "hobbies": []
    }
    sentences = break_into_sentences_with_llm(doc.text)

    pronouns = {'i', 'a', 'us', 'we', 'you', 'he', 'she', 'they', 'me', 'him', 'her', 'them', 'my', 'your', 'his',
                'its', 'our', 'their'}
    for sent in sentences:
        # Analyze sentiment
        print("sent: ", sent)
        sentiment = analyze_sentiment(sent)

        # Adjust for negation
        if "not" in sent.lower() or "n't" in sent.lower():
            sentiment = -1 * analyze_sentiment(sent)  # Invert sentiment if negation is detected

        print(f"Sentence: {sent}, Sentiment: {sentiment}")

        # Process the sentence with SpaCy
        spacy_doc = nlp(sent)  # Create a SpaCy Doc object for the current sentence

        topics = {chunk.text.lower().strip() for chunk in spacy_doc.noun_chunks}
        topics.update(token.lemma_.lower().strip() for token in spacy_doc if
                      token.pos_ in {"VERB", "NOUN", "PROPN", "ADJ"} and len(token.text.split()) == 1)

        # Filter out pronouns
        topics = [topic for topic in topics if topic not in pronouns and len(topic.split()) > 1 or topic.isalpha()]

        # for chunk in spacy_doc.noun_chunks:
        for topic in topics:

            lemmatized_topic = nlp(topic)[0].lemma_

            if topic in pronouns:
                continue

            # Use LLM to categorize the topic
            category = categorize_with_llm(topic)

            print(lemmatized_topic, category)

            # Categorize based on sentiment and LLM response
            # if sentiment > 0.05:
            if category == 'interests':
                profile["interests"].append(lemmatized_topic)
            elif category == 'food':
                profile["foods"].append(lemmatized_topic)
            elif category == 'hobbies':
                profile["hobbies"].append(lemmatized_topic)
            # elif sentiment < -0.05:
            elif category == 'dislikes':
                profile["dislikes"].append(lemmatized_topic)

    profile["interests"] = list(set(profile["interests"]))
    profile["dislikes"] = list(set(profile["dislikes"]))
    profile["foods"] = list(set(profile["foods"]))
    profile["hobbies"] = list(set(profile["hobbies"]))

    return profile