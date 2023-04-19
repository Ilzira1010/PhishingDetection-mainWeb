from flask import Flask, render_template, request, jsonify
import PhishingDetection as pd

app = Flask(__name__)
app.static_folder="static"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    input_text = request.json['input_text']
    result = pd.detectPhising(input_text)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)