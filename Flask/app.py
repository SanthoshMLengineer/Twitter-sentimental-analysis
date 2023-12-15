from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from flask import jsonify

from prediction import prediction_from_model

app = Flask(__name__)
app.secret_key = "super secret key"


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        my_prediction = prediction_from_model(message)
        return render_template('index.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)

