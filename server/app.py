from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, logging as transformers_logging
import warnings

# Suppress warnings from transformers and PyTorch
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Initialize the Flask app
app = Flask(__name__)


# Enable CORS on the Flask app
CORS(app)

# Initialize Flask-RESTPlus API
api = Api(app, version='1.0', title='Sentiment Analysis API',
          description='A simple API to predict sentiment (positive, negative, neutral) of a given text.')

# Define the model input/output formats using Flask-RESTPlus fields
predict_model = api.model('PredictModel', {
    'text': fields.String(required=True, description='The text to analyze for sentiment'),
    'max_len': fields.Integer(description='The maximum length of the text for tokenization', default=64)
})

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.load_state_dict(torch.load('model/sentiment_model.pth'))
model.eval()

# Define a helper function to get prediction insights
def predict_sentiment_with_insights(text, model, tokenizer, max_len=64):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=-1).flatten()
    prediction = torch.argmax(probabilities).item()
    confidence = probabilities[prediction].item()
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10)).item()
    
    sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    predicted_label = sentiment_map[prediction]
    
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "class_probabilities": {sentiment_map[i]: prob.item() for i, prob in enumerate(probabilities)},
        "uncertainty_score": entropy
    }

# Route 1: Basic sentiment prediction
@api.route('/predict')
class Predict(Resource):
    @api.doc(description="Predict sentiment of the given text")
    @api.expect(predict_model, validate=True)
    def post(self):
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({"error": "Text is required"}), 400
        result = predict_sentiment_with_insights(text, model, tokenizer)
        return jsonify(result)

# Route 2: Prediction with custom max length for analysis of longer texts
@api.route('/predict_custom_length')
class PredictCustomLength(Resource):
    @api.doc(description="Predict sentiment with custom max length for tokenization")
    @api.expect(predict_model, validate=True)
    def post(self):
        data = request.get_json()
        text = data.get('text')
        max_len = data.get('max_len', 64)
        if not text:
            return jsonify({"error": "Text is required"}), 400
        result = predict_sentiment_with_insights(text, model, tokenizer, max_len)
        return jsonify(result)

# Route 3: Get raw model output logits for advanced analysis
@api.route('/predict_raw')
class PredictRaw(Resource):
    @api.doc(description="Get raw model output logits for advanced analysis")
    @api.expect(predict_model, validate=True)
    def post(self):
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({"error": "Text is required"}), 400
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.flatten().tolist()
        
        return jsonify({"logits": logits})

# Route 4: Get sentiment prediction for a list of texts
@api.route('/bulk_predict')
class BulkPredict(Resource):
    @api.doc(description="Predict sentiment for a list of texts")
    @api.expect(api.model('BulkPredictModel', {
        'texts': fields.List(fields.String(required=True), description='A list of texts to analyze')
    }), validate=True)
    def post(self):
        data = request.get_json()
        texts = data.get('texts')
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "List of texts is required"}), 400
        
        results = [predict_sentiment_with_insights(text, model, tokenizer) for text in texts]
        return jsonify(results)

# Route 5: Summary of model insights for testing/diagnostics
@api.route('/model_info')
class ModelInfo(Resource):
    @api.doc(description="Get model information for diagnostics")
    def get(self):
        return jsonify({
            "model_name": "DistilBERT Sentiment Analysis Model",
            "tokenizer": "DistilBertTokenizer",
            "max_sequence_length": 64,
            "sentiment_classes": ["negative", "positive", "neutral"],
            "training_note": "This model is fine-tuned for sentiment classification tasks."
        })

# Start the Flask server
if __name__ == '__main__':
    app.run(debug = True)
