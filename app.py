import os
import pandas as pd
from flask import Flask, request, render_template
from model import TextPreprocessor
import task

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    model=task.load_pickle.delay()
    model.wait()
    
    final_features = pd.Series(data=str(request.form['headline']))
    prediction = model.predict(final_features)
        
    output = prediction[0]
    
    return render_template('result.html', prediction=output)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)