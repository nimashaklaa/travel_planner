import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

model_path = './classifier_model'
loaded_tokenizer = BertTokenizer.from_pretrained(model_path)
loaded_model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cpu')  # or 'cuda' if available
loaded_model.to(device)

print("Model loaded successfully!")


def classify_email(email_text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        email_text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

    return "Travel-related" if predicted.item() == 1 else "Non-travel-related"


#### Test Set (Should  be replaced with the actual emails we fetched) ####

df_test = pd.read_csv(
    "test_email_dataset.csv",
    encoding='latin1')

texts_test = list(df_test["Email_Body"])
labels_test = list(df_test["Category"])
labels_test = [1 if label == 'Travel' else 0 for label in labels_test]

labels_predicted = []
travel_related_emails = []  # Initialize the list to store travel-related emails
counter = 0  # Initialize the counter

for x in texts_test:

    result = classify_email(x, loaded_model, loaded_tokenizer, device)
    # print(x,result)
    print(f"{counter + 1}/{len(texts_test)}: {x} -> {result}")  # Display the progress
    if result == "Travel-related":
        labels_predicted.append(1)
        travel_related_emails.append(x)  # Append the email to the list
    else:
        labels_predicted.append(0)
    counter += 1  # Increment the counter


#### Calculate Accuracy ####


def calculate_accuracy(array1, array2):
    # Convert lists to numpy arrays
    array1 = np.array(array1)
    array2 = np.array(array2)

    # Calculate accuracy
    accuracy = np.mean(array1 == array2) * 100  # Percentage of correct predictions
    return accuracy


accuracy = calculate_accuracy(labels_test, labels_predicted)
print(f"Accuracy: {accuracy:.2f}%")