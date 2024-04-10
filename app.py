from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('sentiment_analysis_model.pkl')



from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('sentiment_analysis_model.pkl')  # Load the trained model

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    text = request.form['text']

    # Preprocess the input text if needed

    # Make predictions using the loaded model
    prediction = model.predict([text])[0]

    # Return the prediction to the user
    return render_template('result.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)


