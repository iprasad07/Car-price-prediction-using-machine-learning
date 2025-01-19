from flask import Flask, render_template, request
import pandas as pd
import numpy as  np
import pickle

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
data = pd.read_csv('cleaned_car_data.csv')


@app.route('/')
def index():
    companies = sorted(data['company'].unique())
    car_models = sorted(data['name'].unique())
    year = sorted(data['year'].unique(), reverse=True)
    fuel_type = sorted(data['fuel_type'].unique())
    companies.insert(0, "select Company")
    car_models.insert(0, "select model")
    year.insert(0, "select Year")
    fuel_type.insert(0, "select Fuel Type")
    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           year=year,
                           fuel_type=fuel_type)


@app.route('/predict', methods=["POST"])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))
    print(company, car_model, year, fuel_type,kms_driven)
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
                                            ))
    print(prediction)
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)