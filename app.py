from flask import Flask,render_template,request,redirect,session,url_for
import sys
import pandas as pd
from src.pipeline.prediction import predictPipeline, CustomData
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)
app.secret_key = "secret_key_for_session" # In production, use a secure environment variable

@app.route('/')
def index():
    # If there are results or data in the session, use them and then clear them
    # This allows the "Refresh" button to reset the page to the default state
    results = session.pop('results', None)
    data = session.pop('data', {})
    return render_template('home.html', results=results, data=data)

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return redirect(url_for('index'))
    else:
        try:
            gender = request.form['gender']
            race_ethnicity = request.form['ethnicity']
            parental_level_of_education = request.form['parental_level_of_education']
            lunch = request.form['lunch']
            test_preparation_course = request.form['test_preparation_course']
            reading_score = request.form['reading_score']
            writing_score = request.form['writing_score']
            
            data = CustomData(
                gender = gender,
                race_ethnicity = race_ethnicity,
                parental_level_of_education = parental_level_of_education,
                lunch = lunch,
                test_preparation_course = test_preparation_course,
                reading_score = reading_score,
                writing_score = writing_score
            )
            data_df = data.get_data_as_data_frame()
            pipeline = predictPipeline()
            y_pred = pipeline.predict(data_df)
            
            # Save results and input data to session for redirection (PRG pattern)
            session['results'] = round(y_pred[0], 2)
            session['data'] = request.form
            
            return redirect(url_for('index'))
        except Exception as e:
            raise CustomException(e, sys)
    

if __name__ == '__main__':
    app.run(debug=True)