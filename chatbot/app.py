
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# Define userdata dictionary with Google API key


# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY=userdata.get('google_api_key')

genai.configure(api_key=GOOGLE_API_KEY)


for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

model = genai.GenerativeModel('gemini-pro')


response = model.generate_content("What is the meaning of life?")


to_markdown(response.text)

response.prompt_feedback

from flask import Flask, request, jsonify, render_template



app = Flask(__name__)

# Route to handle incoming messages from the user
@app.route('/message', methods=['POST'])
def handle_message():
    data = request.get_json()
    user_input = data['message']
    # Generate response using the GenerativeModel
    response = model.generate_content(user_input)
    return jsonify({'message': response.text})

# Route to render the chatbot interface
@app.route('/')
def chat_interface():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
