import pandas as pd
from flask import Flask, request, render_template
import pickle
from model import TextPreprocessor

app = Flask(__name__)

model = pickle.load(open('pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    final_features = pd.Series(data=str(request.form['headline']))
    prediction = model.predict(final_features)
    
       
    output = prediction[0]
    
    return render_template('result.html', prediction=output)

if __name__ == "__main__":
    app.run(debug=True)