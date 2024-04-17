# Python file of all tools that can be executed by the agents
import pandas as pd
import os
import pickle
from typing import Annotated

def get_subject_info(file_name: Annotated[str, "The name of the csv file that contains all data."], 
                     subject: Annotated[str, "The name of the subject to extract information for."], 
                     feature: Annotated[str, "The name of the column in the data to extract information for."])->str:
    """
    A function to extract information of a particular feature column for a specific subject from a csv file.
    """
    df = pd.read_csv(os.path.join("data",file_name))
    subject_line = df.loc[df['Name']==subject]
    subject_feature = subject_line[feature].values[0]
    return f"The {feature} for {subject} is {subject_feature}."

def prepare_model_input(file_name,subject):
    """
    A helper function to prepare the data from a subject as input to a machine learning model.
    This function is not registered by the agents.
    """
    df = pd.read_csv(os.path.join("data",file_name))
    features = ['AWD','RWD','Engine Size (l)','Cyl','Horsepower(HP)','City Miles Per Gallon','Highway Miles Per Gallon','Weight','Wheel Base','Len','Width']
    feature_data = df[features]
    type_dummy = pd.get_dummies(df['Type']).astype(int)
    feature_data = pd.concat([feature_data,type_dummy],axis=1)
    subject_data = feature_data.loc[df['Name']==subject]
    return subject_data

def make_price_prediction(file_name: Annotated[str, "The name of the csv file that contains all data."], 
                          subject: Annotated[str, "The name of the subject to predict price for."])->str:
    """
    A function to predict the price of the given subject using a trained machine learning model.
    """
    subject_data = prepare_model_input(file_name,subject)
    X = subject_data.to_numpy()
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    predicted = model.predict(X)[0]
    return f"The predicted value for {subject} is {'%.2f' % (predicted)} according to the ML model."
    

def compute_prediction_change(file_name: Annotated[str, "The name of the csv file that contains all data."],
                              subject: Annotated[str, "The name of the subject to predict price for."],
                              feature: Annotated[str, "The name of the column in the data that changes."],
                              change: Annotated[float,"The amount of change for the specific feature column."]):
    """
    A function to compute the change in the predicted price of the given subject using a trained machine learning model,
    when the value of a speific feature is changed by a certain amount.
    """
    subject_data = prepare_model_input(file_name,subject)
    original_feature = subject_data[feature].values[0]
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    original_predicted = model.predict(subject_data.to_numpy())[0]
    new_feature = original_feature + change
    subject_data[feature] = new_feature
    new_predicted = model.predict(subject_data.to_numpy())[0]
    return f"When the feature {feature} of {subject} changes by {change} from {original_feature} to {new_feature}, the predicted value changes from {'%.2f' % (original_predicted)} to {'%.2f' % (new_predicted)} according to the ML model."