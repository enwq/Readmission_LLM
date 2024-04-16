# Python file of all tools that can be executed by the agents
import pandas as pd
import os
import pickle

def get_subject_info(file_name: str, subject: str, feature: str)->str:
    """
    Use this function to extract information of a particular feature column for a specific subject from a csv file.
    file_name is the name of the csv file that contains all data.
    subject is the name of the subject to extract information for.
    feature is the name of the column in the data to extract information for.
    """
    df = pd.read_csv(os.path.join("data",file_name))
    subject_line = df.loc[df['Name']==subject]
    subject_feature = subject_line[feature].values[0]
    return f"The {feature} for {subject} is {subject_feature}."

def make_price_prediction(file_name: str, subject: str)->str:
    """
    Use this function to predict the price of the given subject using a trained machine learning model.
    file_name is the name of the csv file that contains all data.
    subject is the name of the subject to make prediction for.
    """
    df = pd.read_csv(os.path.join("data",file_name))
    features = ['AWD','RWD','Engine Size (l)','Cyl','Horsepower(HP)','City Miles Per Gallon','Highway Miles Per Gallon','Weight','Wheel Base','Len','Width']
    feature_data = df[features]
    type_dummy = pd.get_dummies(df['Type']).astype(int)
    feature_data = pd.concat([feature_data,type_dummy],axis=1)
    subject_data = feature_data.loc[df['Name']==subject]
    X = subject_data.to_numpy()
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    predicted = model.predict(X)[0]
    return f"The predicted value for {subject} is {'%.2f' % (predicted)} according to the ML model."
    

