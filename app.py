import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

def noInf(value1, value2):
    """
    Takes in two values and divides the first by the second
    If valu2 is 0 then take the value to be 0
    """
    if(value2 == 0):
        return(0)
    else:
        return(value1 / value2)

app = Flask(__name__)
# Put the new model here
model = pickle.load(open('Model/lgb_flask_1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    initial_features = [float(x) for x in request.form.values()]
    final_features = []
    final_features.append(noInf(initial_features[0], initial_features[3]))
    final_features.append(initial_features[5])
    final_features.append(noInf(initial_features[5], initial_features[4]))
    final_features.append(noInf((initial_features[0] + initial_features[1] + initial_features[2]), initial_features[3]))
    final_features.append(noInf(initial_features[5], initial_features[6]))
    final_features.append(initial_features[7])

    final = [np.array(final_features)]
    prediction = model.predict(final)

    output = round(prediction[0], 4)

    return render_template('index.html', prediction_text='Player\'s percentile is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
