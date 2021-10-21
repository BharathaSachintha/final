
from flask import Flask, render_template, request, redirect, url_for, session
from flask.wrappers import JSONMixin
import mysql.connector
import re
import pickle
import numpy as np

from datetime import datetime
import pandas as pd
import json



# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database = "diabates"

)

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', username = str(session['username']))
    return render_template('index.html')


@app.route('/signup',methods=['GET','POST'])
def signup():
    msg = ''
    if(request.method == "POST"):
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        cursor = mydb.cursor()
        query = 'SELECT * FROM users WHERE email = %s'
        values = (email,)
        cursor.execute(query, values)
        user = cursor.fetchone()
        if user:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            query ='INSERT INTO users(username, email, password) VALUES (%s, %s, %s)'
            values = (username, email, password)
            cursor.execute(query, values)
            mydb.commit()
            session['username'] = str(username)
            msg = 'You have successfully registered!'
            print(cursor.rowcount, 'record inserted')
            print("session username is : " + str(session['username']))
            session.pop('uid', None)
            session.pop('username', None)
            return render_template('login.html')     
        return render_template('index.html')
    return render_template('signup.html')
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if(request.method == "POST"):
        email = request.form['email']
        password = request.form['password']
        cursor = mydb.cursor()
        query = 'SELECT * FROM users WHERE email = %s AND password = %s'
        values = (email, password,)
        cursor.execute(query, values)
        user = cursor.fetchone()
        if user:
            session['uid'] = str(user[0])
            session['username'] = user[1]
            return render_template('index.html', username=str(session['username']))
        else:
            msg = 'Incorrect username/password!'
            return render_template('login.html', msg = msg)
    return render_template('login.html')

@app.route('/directrecords', methods=['GET', 'POST'])
def directrecords():
    if request.method == 'POST':
        complications = ""
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        cuid = str(session['uid'])
        if my_prediction == [0]:
            res1 = "Negative"

        if my_prediction == [1]:
            res1 = "Positive"
            if int(glucose) < 125:
                complications = "Hypoglycimia"
            if int(glucose) > 125:
                complications = "Hyperglycimia, Macrovascular, Microvascular"
        
        now = datetime.now()
        dt = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt)
        cursor = mydb.cursor()
        query1 ='INSERT INTO tests(uid, preg, glucose, bp, st, insulin, bmi, dpf, age, testtime, result) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
        values1 = (cuid, str(preg), str(glucose), str(bp), str(st), str(insulin), str(bmi), str(dpf), str(age), str(dt), str(res1))
        cursor.execute(query1, values1)
        mydb.commit()

        return render_template('result.html', prediction=my_prediction, complications = complications)
    return render_template('directrecords.html')

@app.route('/questionaire', methods=['GET', 'POST'])
def questionaire():
    return render_template('questionaire.html')


# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------


    

# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.route('/meet', methods=['GET', 'POST'])
def meet():
    if request.method == 'POST':
        lix: str = request.values['li']
        lix2: str = json.loads(lix)
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb=gnb.fit(X,np.ravel(y))
        
        # calculating accuracy-------------------------------------------------------------------
        from sklearn.metrics import accuracy_score
        y_pred=gnb.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
        # -----------------------------------------------------
        psymptoms = lix2
       
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = gnb.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        
        
        if (h=='yes'):
            print(disease[a])
            return str(disease[a])
        
        else:
            print("Not Found")
            return "Nothing"

        
         
    return render_template('meet.html')



@app.route('/meetresult', methods=['GET', 'POST'])
def meetresult():
    return render_template('meetresult.html')

@app.route('/doctors', methods=['GET', 'POST'])
def doctors():
    return render_template('doctors.html')

@app.route('/qresult', methods=['GET', 'POST'])
def qresult():
    if request.method == 'POST':
        dataxlengeth = request.values['dataxlengeth']
        datax = request.values['datax']
        print("------------> "+str(datax))
        print("------------> "+str(dataxlengeth))
        return render_template('qresult.html', datax = datax)
    return render_template('qresult.html')

@app.route('/history')
def history():
    if 'uid' in session:
        cursor = mydb.cursor()
        query3 = 'select * from tests WHERE uid = %s'
        values3 = (session['uid'],)
        cursor.execute(query3, values3)
        data = cursor.fetchall()
        print(str(data))
        return render_template('history.html', data = data)
    return render_template('login.html')

@app.route('/logout')
def logout():
   session.pop('uid', None)
   session.pop('username', None)
   return redirect(url_for('index'))




if __name__ == '__main__':
  app.run()
 