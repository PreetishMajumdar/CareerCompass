from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data and convert to float
        cgpa = float(request.form.get('cgpa'))
        iq = float(request.form.get('iq'))
        profile_score = float(request.form.get('profile_score'))
        proficient_lang = float(request.form.get('proficient_lang'))
        extracurricular = float(request.form.get('extracurricular'))
        internship = float(request.form.get('internship'))
        projects = float(request.form.get('projects'))
        softskills = float(request.form.get('softskills'))
        certifications = float(request.form.get('certifications'))
        interviews = float(request.form.get('interviews'))
        github_repositories = float(request.form.get('github_repositories'))
        workshop = float(request.form.get('workshop'))
        socialwork = float(request.form.get('socialwork'))
        aptitudetest = float(request.form.get('aptitudetest'))

        # Create a numpy array with all the input data
        input_query = np.array([[cgpa, iq, profile_score, proficient_lang, extracurricular, internship, projects,
                                 softskills, certifications, interviews, github_repositories, workshop, socialwork,
                                 aptitudetest]])

        # Make prediction
        result = model.predict(input_query)[0]

        # Return the result as JSON
        return jsonify({'placement': str(result)})

    except ValueError as e:
        # Handle the case where conversion to float fails
        return jsonify({'error': 'Invalid input. Ensure all fields are numeric.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
