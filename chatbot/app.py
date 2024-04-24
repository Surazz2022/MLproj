from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dpreprocessor import dataset 

app = Flask(__name__)

# Load pre-trained GPT model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate response using the pre-trained model
def generate_response(user_input, max_length=50):
    input_ids = tokenizer.encode(user_input, return_tensors="pt", max_length=max_length, truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Route to handle incoming messages from the user
@app.route('/message', methods=['POST'])
def handle_message():
    data = request.get_json()
    user_input = data['message']
    response = generate_response(user_input)
    return jsonify({'message': response})

# Route to render the chatbot interface
@app.route('/')
def chat_interface():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
