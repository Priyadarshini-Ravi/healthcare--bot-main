import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import json
import warnings
import streamlit as st
from streamlit_lottie import st_lottie
warnings.filterwarnings("ignore", category=DeprecationWarning)


st.set_page_config(
    page_title="HealthCare ChatBot",
    page_icon="🧊"
)


def get(path: str):
    with open(path, 'r') as f:
        return json.load(f)


home_path = get('./home.json')
login_path = get('./Login.json')

training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y


reduced_data = training.groupby(training['prognosis']).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)


clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print(scores.mean())


model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days, severity_dict):
    total_severity = sum(severity_dict[item] for item in exp)
    avg_severity = (total_severity * days) / (len(exp) + 1)

    if avg_severity > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."


# Function to load severity data from CSV
def getSeverityDict():
    severityDictionary = {}
    with open('MasterData/severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row:  # Check if row is not empty
                if len(row) >= 2:  # Ensure the row has at least two elements
                    severityDictionary[row[0]] = int(row[1])
                else:
                    print("Issue with row:", row)
            else:
                print("Empty row detected")
    return severityDictionary

# Function to load description data from CSV


def getDescription():
    description_list = {}
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]
    return description_list

# Function to load precaution data from CSV


def getprecautionDict():
    precautionDictionary = {}
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    return precautionDictionary

# Function to get user information using Streamlit


def getInfo():
    st.title("HealthCare ChatBot")
    name = st.text_input("Your Name?")
    st.write(f"Hello, {name}")

# Function to check for patterns in user input


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return True, pred_list
    else:
        return False, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_diagnosis(clf, feature_names, user_symptoms, num_days, severity_dict, description_dict, precaution_dict):
    tree_ = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    def recurse(node, depth, user_symptoms, num_days):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name in user_symptoms:
                val = 1
            else:
                val = 0

            if val <= threshold:
                recurse(tree_.children_left[node],
                        depth + 1, user_symptoms, num_days)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node],
                        depth + 1, user_symptoms, num_days)
        else:
            present_disease = print_disease(tree_.value[node])
            symptoms_exp = []  # Use Streamlit input elements to get symptoms
            second_prediction = sec_predict(symptoms_exp)
            condition_result = calc_condition(
                symptoms_exp, num_days, severity_dict)

            # Display diagnosis info using Streamlit interface
            if present_disease[0] == second_prediction[0]:
                st.write("You may have ", present_disease[0])
                st.write(description_dict[present_disease[0]])
            else:
                st.write("You may have ",
                         present_disease[0], "or ", second_prediction[0])
                st.write(description_dict[present_disease[0]])
                st.write(description_dict[second_prediction[0]])

            precution_list = precaution_dict[present_disease[0]]
            st.write("Take following measures : ")
            for i, j in enumerate(precution_list):
                st.write(i + 1, ")", j)

    recurse(0, 1, user_symptoms, num_days)


# Load data and necessary information
severityDictionary = getSeverityDict()
description_list = getDescription()
precautionDictionary = getprecautionDict()
# Load your classifier
clf = DecisionTreeClassifier()  # You need to load your trained classifier here

# Fit your classifier with training data
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
clf = clf.fit(x_train, y_train)

# Initiate Streamlit app


def main():
    st.title("HealthCare ChatBot")

    menu = ["Home", "Login"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st_lottie(home_path)
        # Your home page content goes here

    elif choice == "Login":
        st_lottie(login_path)

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.checkbox("Login"):
            if username == "Priyadharshini" and password == "samplepass":
                st.success("Logged In as {}".format(username))
                
                # Your application code goes here
                user_input = st.text_input(
                    "Enter your symptoms separated by comma")  # Get user symptoms
                num_days = st.number_input(
                    "From how many days?", min_value=1, key='num_days')  # Get number of days

                if st.button("Diagnose"):  # Button to trigger diagnosis
                    # Split user input into list of symptoms
                    user_symptoms = user_input.split(",")

                    # Obtain feature names from the training dataset columns
                    # Assuming the last column is the target
                    feature_names = list(training.columns[:-1])

                    # Call tree_diagnosis with all the necessary parameters
                    tree_diagnosis(clf, feature_names, user_symptoms, num_days, severityDictionary, description_list,
                                   precautionDictionary)
            else:
                st.warning("Incorrect Username/Password")


if __name__ == "__main__":
    main()
