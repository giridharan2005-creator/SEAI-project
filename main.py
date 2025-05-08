from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import cohere

# Initialize Flask app
app = Flask(__name__)

# Initialize Cohere client
co = cohere.Client('4cTF1EcMneLsoDX6UEq5TdUfv43LyYQZxirGCbse')  

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load trained model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Symptom dictionary and disease list
symptoms_dict = { ... }  # ⬅️ Your full symptoms_dict goes here
diseases_list = { ... }  # ⬅️ Your full diseases_list goes here

# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms" or not symptoms.strip():
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)

        # Process symptoms
        user_symptoms = [s.strip().lower() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

        # Predict disease
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        my_precautions = []
        for i in precautions[0]:
            my_precautions.append(i)

        # Use Cohere to summarize symptoms
        cohere_prompt = f"A patient reports the following symptoms: {', '.join(user_symptoms)}. Provide a clear and concise summary of their condition."
        try:
            cohere_response = co.generate(
                model='command-r',
                prompt=cohere_prompt,
                max_tokens=100,
                temperature=0.6
            )
            llm_summary = cohere_response.generations[0].text.strip()
        except Exception as e:
            llm_summary = f"Could not generate summary: {str(e)}"

        return render_template('index.html',
                               predicted_disease=predicted_disease,
                               dis_des=llm_summary,
                               my_precautions=my_precautions,
                               medications=medications,
                               my_diet=rec_diet,
                               workout=workout)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
