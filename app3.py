from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from geopy.distance import geodesic 

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
city_center_coordinates = [55.7522, 37.6156]

def get_azimuth(latitude, longitude):
 
    rad = 6372795

    llat1 = city_center_coordinates[0]
    llong1 = city_center_coordinates[1]
    llat2 = latitude
    llong2 = longitude

    lat1 = llat1*math.pi/180.
    lat2 = llat2*math.pi/180.
    long1 = llong1*math.pi/180.
    long2 = llong2*math.pi/180.

    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
    x = sl1*sl2+cl1*cl2*cdelta
    ad = math.atan2(y,x)

    x = (cl1*sl2) - (sl1*cl2*cdelta)
    y = sdelta*cl2
    z = math.degrees(math.atan(-y/x))

    if (x < 0):
        z = z+180.

    z2 = (z+180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi))) )
    angledeg = (anglerad2*180.)/math.pi
    
    return round(angledeg, 2)

@app.route('/')
def home():
	return render_template('2.html')

@app.route('/predict',methods=['POST'])
def predict():
    Material=int(request.form['Material']),
    Floor=int(request.form['Floor']),
    Total_floor=int(request.form['Total_floor']),
    Rooms=int(request.form['Rooms']),
    Type=int(request.form['Type']),
    Administrative_distrinct=int(request.form['Administrative_distrinct']),
    Area=int(request.form['Area']),
    Distance=int(request.form['Distance']),
    Azimut=int(request.form['Azimut']),
	#df = pd.DataFrame([[Material,Floor,Total_floor,Rooms,Type,Administrative_distrinct,Area,Distance,Azimut]],columns=['Material','Floor','Total_floor','Rooms','Type','Administrative_distrinct','Area','Distance','Azimut'])
    features_0 = np.array([Material,Floor,Total_floor,Rooms,Type,Administrative_distrinct,Area,Distance,Azimut])
    #prediction = model.predict(features)
    #output = round(prediction[0],-3)
    #prediction = model.predict(df).round(0)
	#price = prediction * df['Area'][0]
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    for_test = np.array([final_features])
    prediction_0 = model.predict(final_features)
    prediction = prediction_0 * features_0[6]
    output = round(prediction[0],-3)
    return render_template('2.html', prediction_text='Предсказанная цена (в рублях): {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
