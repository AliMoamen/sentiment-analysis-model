import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Load the saved model's state_dict
model.load_state_dict(torch.load('model/sentiment_model.pth'))
model.eval()

# Save the tokenizer to a .pkl file
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Save the model to a .pkl file (you can save only the state_dict if you prefer)
with open('model.pkl', 'wb') as f:
    pickle.dump(model.state_dict(), f)  # Saving only the model weights
    # Alternatively, you can pickle the entire model like this:
    # pickle.dump(model, f)

print("Tokenizer and model saved as .pkl files")
