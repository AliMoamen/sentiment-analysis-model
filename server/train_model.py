import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import warnings
from torch.cuda.amp import GradScaler, autocast  # For mixed precision

# Suppress the weight initialization warning for clarity
warnings.filterwarnings('ignore', category=UserWarning, message="Some weights of .* were not initialized from the model checkpoint")

# Define a custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts.iloc[index]
        label = self.labels.iloc[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = SentimentDataset(
        texts=df['text'],
        labels=df['sentiment'],
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

def train_model(model, data_loader, optimizer, loss_fn, device, scheduler=None, num_epochs=3, grad_accum_steps=2):
    model = model.train()
    scaler = GradScaler()  # For mixed precision

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for step, batch in enumerate(tqdm(data_loader)):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # Enable mixed precision with autocast
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels) / grad_accum_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:  # Update weights every `grad_accum_steps`
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # For accuracy calculation
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            total_loss += loss.item() * grad_accum_steps  # Correct for gradient accumulation

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = correct_predictions.double() / total_predictions

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Loss: {avg_loss}, Accuracy: {avg_accuracy}")
        if scheduler:
            scheduler.step()

    # Save the model and tokenizer after training
    model.save_pretrained('model')
    tokenizer.save_pretrained('model')

def evaluate_model(model, data_loader):
    model = model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(model.device, non_blocking=True),
                'attention_mask': batch['attention_mask'].to(model.device, non_blocking=True)
            }
            with autocast():  # Use mixed precision in evaluation for speed
                outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(batch['label'].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, f1

if __name__ == '__main__':
    # Load train and test datasets
    train_df = pd.read_csv('data/train.csv', encoding='ISO-8859-1')
    test_df = pd.read_csv('data/test.csv', encoding='ISO-8859-1')

    # Prepare datasets by mapping sentiments to integers
    train_df = train_df[['text', 'sentiment']].dropna()
    test_df = test_df[['text', 'sentiment']].dropna()
    train_df['sentiment'] = train_df['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})
    test_df['sentiment'] = test_df['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})

    # Set up tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    # Set hyperparameters and create data loaders
    BATCH_SIZE = 16
    MAX_LEN = 64
    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    loss_fn = CrossEntropyLoss()

    # Train the model
    train_model(model, train_data_loader, optimizer, loss_fn, device, scheduler=scheduler, num_epochs=3, grad_accum_steps=4)

    # Evaluate the model
    accuracy, f1 = evaluate_model(model, test_data_loader)
    print(f"Accuracy: {accuracy}, F1-Score: {f1}")

    # Load the trained model and tokenizer (if needed for inference)
    model = DistilBertForSequenceClassification.from_pretrained('model', num_labels=3)
    tokenizer = DistilBertTokenizer.from_pretrained('model')
