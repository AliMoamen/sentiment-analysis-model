# Sentiment Analysis Web App

This project provides a sentiment analysis web app built using React for the frontend and Flask for the backend. It utilizes the DistilBERT model for predicting the sentiment of input text as positive, negative, or neutral. The app supports both single text sentiment prediction and bulk predictions. Additionally, it offers options to analyze sentiment with a custom tokenization length.

## Features

- **Sentiment Prediction**: Predict sentiment (positive, negative, neutral) for a given text.
- **Custom Tokenization Length**: Users can specify the maximum length for tokenizing the text.
- **Bulk Prediction**: Analyze sentiment for a list of texts.
- **Model Insights**: Includes confidence score, class probabilities, and uncertainty score.
- **Flask API**: Exposes RESTful endpoints to interact with the sentiment analysis model.
- **React Frontend**: User-friendly UI to input text and view results.

## Model Performance

The sentiment analysis model was fine-tuned using a dataset of **27,840 social media posts** and tested on **3,550 social media posts**, achieving the following performance metrics:

- **Accuracy**: 79.9%
- **F1-Score**: 0.80

These results demonstrate the model's effectiveness in predicting sentiment across various types of social media content.

## Installation

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AliMoamen/sentiment-analysis-model.git
   cd sentiment-analysis-model
   ```

2. **Install dependencies and start the app**:

   ```bash
   npm install
   npm start
   ```

3. **Navigate to the app**:
   Open your browser and go to `http://localhost:5173/` to view the app.

## API Endpoints

The server is running on `http://127.0.0.1:5000` by default.

### 1. `/predict` (POST)

Predict the sentiment of the provided text.

- **Request Body**:

  ```json
  {
    "text": "This is an example text."
  }
  ```

- **Response**:
  ```json
  {
    "predicted_label": "positive",
    "confidence": 0.85,
    "class_probabilities": {
      "negative": 0.1,
      "positive": 0.85,
      "neutral": 0.05
    },
    "uncertainty_score": 0.23
  }
  ```

### 2. `/predict_custom_length` (POST)

Predict sentiment with a custom maximum length for tokenization.

- **Request Body**:

  ```json
  {
    "text": "This is an example text.",
    "max_len": 128
  }
  ```

- **Response**: Same as `/predict`.

### 3. `/predict_raw` (POST)

Get the raw model logits (useful for advanced analysis).

- **Request Body**:

  ```json
  {
    "text": "This is an example text."
  }
  ```

- **Response**:
  ```json
  {
    "logits": [-0.345, 0.652, 0.123]
  }
  ```

### 4. `/bulk_predict` (POST)

Predict sentiment for a list of texts.

- **Request Body**:

  ```json
  {
    "texts": [
      "I love this product!",
      "This is the worst thing I have ever used.",
      "It's okay, not great but not bad."
    ]
  }
  ```

- **Response**:
  ```json
  [
    {
      "predicted_label": "positive",
      "confidence": 0.98,
      "class_probabilities": {
        "negative": 0.02,
        "positive": 0.98,
        "neutral": 0
      },
      "uncertainty_score": 0.05
    },
    {
      "predicted_label": "negative",
      "confidence": 0.92,
      "class_probabilities": {
        "negative": 0.92,
        "positive": 0.05,
        "neutral": 0.03
      },
      "uncertainty_score": 0.12
    },
    {
      "predicted_label": "neutral",
      "confidence": 0.55,
      "class_probabilities": {
        "negative": 0.2,
        "positive": 0.25,
        "neutral": 0.55
      },
      "uncertainty_score": 0.67
    }
  ]
  ```

### 5. `/model_info` (GET)

Get information about the trained sentiment model.

- **Response**:
  ```json
  {
    "model_name": "DistilBERT Sentiment Analysis Model",
    "tokenizer": "DistilBertTokenizer",
    "max_sequence_length": 64,
    "sentiment_classes": ["negative", "positive", "neutral"],
    "training_note": "This model is fine-tuned for sentiment classification tasks."
  }
  ```
