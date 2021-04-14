from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
#model = pickle.load(open('model.pkl','rb'))

@app.route('/home')
@app.route('/')
def home():
	return render_template('1.html')
    
@app.route('/ocenka')
def predict_page():
	return render_template('2.html')
    
@app.route('/graphics')
def graphics_page():
	return render_template('3.html')

#@app.route('/predict',methods=['POST'])
#def predict():
    #Material=int(request.form['Material']),
    #Floor=int(request.form['Floor']),
    #Total_floor=int(request.form['Total_floor']),
    #Rooms=int(request.form['Rooms']),
    #Type=int(request.form['Type']),
    #Administrative_distrinct=int(request.form['Administrative_distrinct']),
    #Area=int(request.form['Area']),
    #Distance=int(request.form['Distance']),
    #Azimut=int(request.form['Azimut']),
    #features_0 = np.array([Material,Floor,Total_floor,Rooms,Type,Administrative_distrinct,Area,Distance,Azimut])
   # int_features = [int(x) for x in request.form.values()]
   # final_features = [np.array(int_features)]
   # for_test = np.array([final_features])
   # prediction_0 = model.predict(final_features)
   # prediction = prediction_0 * features_0[6]
   # output = round(prediction[0],-3)
   # return render_template('2.html', prediction_text='Предсказанная цена (в рублях): {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
