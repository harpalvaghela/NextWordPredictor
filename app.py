from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from flask import render_template
from flask_cors import CORS
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def predict_next_words(text, num_predictions=3):
    num_beams = max(num_predictions, 3)
    
    input_ids = tokenizer.encode(text, return_tensors="pt")
    # Note: Consider moving model to eval mode with model.eval()
    
    with torch.no_grad():
        # Generate sequences and also return the logits by setting return_dict_in_generate=True
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 1,
            num_return_sequences=num_predictions,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    logits = outputs.scores[-1]  # Get logits of the last token for each sequence
    softmax_probs = F.softmax(logits, dim=-1)  # Compute softmax to get probabilities

    predicted_words_with_confidence = []
    for i, output in enumerate(outputs.sequences):
        predicted_token_id = output[-1].item()  # Get the last token id
        predicted_word = tokenizer.decode(predicted_token_id)
        confidence_score = softmax_probs[i, predicted_token_id].item()  # Confidence score
        
        predicted_words_with_confidence.append({
            "word": predicted_word.strip(),
            "confidence": confidence_score
        })
    
    return predicted_words_with_confidence


def predict_next_sentence(text, max_length=50):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )

    generated_sequence = outputs[0]
    predicted_sentence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    # New logic to end the sentence after it's "finished"
    end_punctuation = {'.', '!', '?'}
    for i, char in enumerate(predicted_sentence[len(text):], start=len(text)):
        if char in end_punctuation:
            # Stop at the first sentence-ending punctuation after the input text
            return predicted_sentence[:i+1]
    
    return predicted_sentence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    num_predictions = data.get('num_predictions', 3)
    predictions = predict_next_words(text, num_predictions)
    return jsonify({"predictions": predictions})

@app.route('/api/v1/predict/sentence', methods=['POST'])
def predict_sentence():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        predicted_sentence = predict_next_sentence(text)
        return jsonify({"predicted_sentence": predicted_sentence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
