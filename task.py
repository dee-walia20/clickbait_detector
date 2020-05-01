import os
import celery
import pickle
from model import TextPreprocessor
app=celery.Celery('sample')

app.conf.update(BROKER_url=os.environ['REDIS_URL'],
                CELERY_RESULT_BACKEND=os.environ['REDIS_URL'])
@app.task
def load_pickle():
    model = pickle.load(open('pipeline.pkl', 'rb'))
    return model                 
