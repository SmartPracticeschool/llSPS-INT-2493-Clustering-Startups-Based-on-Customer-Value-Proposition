import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
startup= Flask(__name__)
model = load('kmeans1.save')
@startup.route('/')
def home():
    return render_template('new.html')
@startup.route('/ymeans',methods=['POST'])
def y_means():
    prediction = model.predict(y_means)
    print(prediction)
    output=prediction[0]
    return render_template('new.html', prediction_text='Prediction'.format(output))
@startup.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.ymean([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)
if __name__ == "__main__":
    startup.run(debug=True)











