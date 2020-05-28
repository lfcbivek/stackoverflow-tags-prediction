from flask import Flask,request,render_template
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import stochastic_gradient
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer



stop_words = stopwords.words('english')

import re
app = Flask(__name__)


def preprocess(text):
    pattern1 = re.compile('[^0-9a-z#+_]')
    pattern2 = re.compile('[/(){}\[\]\|@,;]')
    text = pattern1.sub(' ',text)
    text = pattern2.sub(" ",text)
    text = ' '.join([word.lower() for word in text.split() if not word in stop_words])
    return text
@app.route("/",methods=['GET','POST'])
def view():
    
    if request.method == 'POST':
        user_query = request.form['query']
        
        if user_query == '':
            return render_template('index.html')
        text = preprocess(user_query)
        
        pipe = joblib.load('model.joblib')
        input_data = []
        input_data.append(text)
        predicted_value= pipe.predict(input_data)[0]
        context = {
            'prediction': predicted_value
        }
        
        return render_template('index.html',context =context)
    return render_template('index.html')




if __name__ == "__main__":
    app.run(debug=True)