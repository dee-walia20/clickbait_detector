import pandas as pd
from flask import Flask, request, render_template
import pickle
from model import TextPreprocessor
import time

app = Flask(__name__)
t1=time.perf_counter()
model = pickle.load(open('pipeline.pkl', 'rb'))
t2=time.perf_counter()
print("Time after loading pickle", t2-t1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'.'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    t3=time.perf_counter()
    final_features = pd.Series(data=str(request.form['headline']))
    prediction = model.predict(final_features)
    t4=time.perf_counter()
    print("Time after prediction", t4-t3)
    
    output = prediction[0]
    print(final_features)
    print("\n",prediction)

    return render_template('result.html', prediction=output)

if __name__ == "__main__":
    app.run(debug=True)