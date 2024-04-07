from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from flask import render_template

app = Flask(__name__)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def predict_next_words(text, num_predictions=3):
    num_beams = max(num_predictions, 3)
    
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 1,
            num_return_sequences=num_predictions,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            early_stopping=True
        )
    
    predicted_words = []
    for output in outputs:
        predicted_token_id = output[-1]
        predicted_word = tokenizer.decode(predicted_token_id)
        predicted_words.append(predicted_word.strip())
    
    return predicted_words

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    num_predictions = data.get('num_predictions', 3)
    predictions = predict_next_words(text, num_predictions)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
