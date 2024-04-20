from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Preprocessing
# Handle missing values if any
data.dropna(inplace=True)

# Remove 'Person ID', 'Occupation', 'Gender', and 'BMI Category' columns
data.drop(columns=['Person ID', 'Occupation', 'Gender', 'BMI Category'], inplace=True)

# Rearrange the column order to match the HTML form
data = data[['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Systolic', 'Diastolic', 'Heart Rate', 'Daily Steps', 'Sleep Disorder', 'Insomnia']]

# Separate features and target variables
X = data.drop(columns=['Sleep Disorder', 'Insomnia'])
y_disorder = data['Sleep Disorder']
y_insomnia = data['Insomnia']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split with stratified sampling for both disorders
X_train_disorder, X_test_disorder, y_train_disorder, y_test_disorder = train_test_split(X_scaled, y_disorder, test_size=0.3, random_state=42, stratify=y_disorder)
X_train_insomnia, X_test_insomnia, y_train_insomnia, y_test_insomnia = train_test_split(X_scaled, y_insomnia, test_size=0.3, random_state=42, stratify=y_insomnia)

# Feature selection and training for Sleep Disorder
model_disorder = LogisticRegression(solver='liblinear')
rfe_disorder = RFE(model_disorder, n_features_to_select=9)
rfe_disorder.fit(X_train_disorder, y_train_disorder)
selected_feature_indices_disorder = rfe_disorder.support_
X_train_selected_disorder = X_train_disorder[:, selected_feature_indices_disorder]
log_reg_disorder = LogisticRegression(solver='liblinear')
log_reg_disorder.fit(X_train_selected_disorder, y_train_disorder)

# Hyperparameter tuning for Sleep Disorder
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_disorder = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
grid_search_disorder.fit(X_train_selected_disorder, y_train_disorder)
best_C_disorder = grid_search_disorder.best_params_['C']

# Prediction for Sleep Disorder
y_pred_disorder = log_reg_disorder.predict(X_test_disorder[:, selected_feature_indices_disorder])
accuracy_disorder = accuracy_score(y_test_disorder, y_pred_disorder)
classification_rep_disorder = classification_report(y_test_disorder, y_pred_disorder, output_dict=True)

# Feature selection and training for Insomnia
model_insomnia = LogisticRegression(solver='liblinear')
rfe_insomnia = RFE(model_insomnia, n_features_to_select=9)
rfe_insomnia.fit(X_train_insomnia, y_train_insomnia)
selected_feature_indices_insomnia = rfe_insomnia.support_
X_train_selected_insomnia = X_train_insomnia[:, selected_feature_indices_insomnia]
log_reg_insomnia = LogisticRegression(solver='liblinear')
log_reg_insomnia.fit(X_train_selected_insomnia, y_train_insomnia)

# Hyperparameter tuning for Insomnia
grid_search_insomnia = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
grid_search_insomnia.fit(X_train_selected_insomnia, y_train_insomnia)
best_C_insomnia = grid_search_insomnia.best_params_['C']

# Prediction for Insomnia
y_pred_insomnia = log_reg_insomnia.predict(X_test_insomnia[:, selected_feature_indices_insomnia])
accuracy_insomnia = accuracy_score(y_test_insomnia, y_pred_insomnia)
classification_rep_insomnia = classification_report(y_test_insomnia, y_pred_insomnia, output_dict=True)

@app.route('/')
def home():
    # Render index.html template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=="POST":
        age=request.form.get('age')
        sleepDuration=request.form.get('sleepDuration')
        qualityOfSleep=request.form.get('qualityOfSleep')
        physicalActivityLevel=request.form.get('physicalActivityLevel')
        stressLevel=request.form.get('stressLevel')
        systolic=request.form.get('systolic')
        diastolic=request.form.get('diastolic')
        heartRate=request.form.get('heartRate')
        dailySteps=request.form.get('dailySteps')
        new_data = np.array([[age, sleepDuration,qualityOfSleep ,physicalActivityLevel ,stressLevel, systolic, diastolic, heartRate, dailySteps]])
        new_data_scaled = scaler.transform(new_data)
        prediction_disorder = log_reg_disorder.predict(new_data_scaled[:, selected_feature_indices_disorder])
        if prediction_disorder == 0:
            return redirect(url_for('result', result='No Sleep Disorder', accuracy=accuracy_disorder, precision=classification_rep_disorder['0']['precision'], recall=classification_rep_disorder['0']['recall']))
        else:
            prediction_insomnia = log_reg_insomnia.predict(new_data_scaled[:, selected_feature_indices_insomnia])
            if prediction_insomnia == 0:
                return redirect(url_for('result', result='Sleep Apnea', accuracy=accuracy_insomnia, precision=classification_rep_insomnia['0']['precision'], recall=classification_rep_insomnia['0']['recall']))
            else:
                return redirect(url_for('result', result='Insomnia', accuracy=accuracy_insomnia, precision=classification_rep_insomnia['1']['precision'], recall=classification_rep_insomnia['1']['recall']))

@app.route('/result')
def result():
    result = request.args.get('result')
    accuracy = request.args.get('accuracy')
    precision = request.args.get('precision')
    recall = request.args.get('recall')
    return render_template('result.html', result=result, accuracy=accuracy, precision=precision, recall=recall)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
