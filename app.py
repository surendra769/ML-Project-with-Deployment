import numpy as np
from flask import Flask, request,  jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = float(model.predict(final_features))
    #prediction = float(prediction)

    output = round(prediction, 2)
    #output = prediction
    return render_template('index.html', prediction_text = "Employee Salary will be around  \u20B9 {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)

                           


