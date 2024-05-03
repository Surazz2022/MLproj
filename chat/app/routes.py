from flask import Blueprint, request, jsonify
from src.chatbot import Chatbot

bp = Blueprint('main', __name__)

chatbot = Chatbot()

@bp.route('/chat', methods=['POST'])
def handle_chat():
    user_input = request.form['input']
    response = chatbot.process_input(user_input)
    return jsonify({'response': response})