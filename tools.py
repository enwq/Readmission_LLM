# Python file of all tools that can be executed by the assistant
import pandas as pd
import os
import pickle
from typing import Annotated

def get_admission_info(admission_id: Annotated[int, "The ID of the hospital admission to extract information for."], 
                     feature: Annotated[str, "Name of the feature to extract information for the admission."])->str:
    """
    A function to extract information of a particular feature column for a hospital admission specified by its ID from a csv data file.
    Descriptions of all possible features to extract are provided below.
    'LOS' is the total length of stay.
    'blood' equals to 1 if the patient is diagnosed with blood and blood-forming organs diseases.
    'circulatory' equals to 1 if the patient is diagnosed with circulatory system diseases.
    'congenital' equals to 1 if the patient is diagnosed with congenital anomalies.
    'digestive' equals to 1 if the patient is diagnosed with digestive system diseases.
    'endocrine' equals to 1 if the patient is diagnosed with endocrine, nutritional and metabolic diseases, and immunity disorders.
    'genitourinary' equals to 1 if the patient is diagnosed with henitourinary system diseases.
    'infectious' equals to 1 if the patient is diagnosed with infectious and parasitic diseases.
    'injury' equals to 1 if the patient is diagnosed with injury and poisoning.
    'mental' equals to 1 if the patient is diagnosed with mental disorders.
    'muscular' equals to 1 if the patient is diagnosed with musculoskeletal system and connective tissue diseases.
    'neoplasms' equals to 1 if the patient is diagnosed with neoplasms.
    'nervous' equals to 1 if the patient is diagnosed with nervous system and sense organs diseases.
    'respiratory' equals to 1 if the patient is diagnosed with respiratory system diseases.
    'skin' equals to 1 if the patient is diagnosed with skin and subcutaneous tissue diseases.
    'supp1' equals to 1 if the patient satisfies supplementary classification of external causes of injury and poisoning.
    'supp2' equals to 1 if the patient satisfies supplementary classification of factors influencing health status and contact with health services.
    'symtoms_signs' equals to 1 if the patient shows symptoms, signs and ill-defined conditions.
    'GENDER' equals to 1 if the patient is male.
    'age' is the age of the admitted patient.
    'ICU' equals to 1 if the patient is admitted to the ICU.
    'ICU_LOS' is the total length of stay in ICU.
    'ADM_ELECTIVE' equals to 1 if this admission is elective.
    'ADM_EMERGENCY' equals to 1 if this admission is emergency.
    'ADM_URGENT' equals to 1 if this admission is urgent.
    'INS_Government' equals to 1 if the insurance of this admission is government.
    'INS_Medicaid' equals to 1 if the insurance of this admission is Medicaid.
    'INS_Medicare' equals to 1 if the insurance of this admission is Medicare.
    'INS_Private' equals to 1 if the insurance of this admission is private.
    'INS_Self Pay' equals to 1 if the insurance of this admission is self pay.
    'ETH_ASIAN' equals to 1 if the admitted patient is Asian.
    'ETH_BLACK/AFRICAN AMERICAN' equals to 1 if the admitted patient is Black/African American.
    'ETH_HISPANIC/LATINO' equals to 1 if the admitted patient is Hispanic/Latino.
    'ETH_OTHER/UNKNOWN' equals to 1 if ethinicity of the admitted patient is Other/Unknown.
    'ETH_WHITE' equals to 1 if the admitted patient is White.
    'MAR_MARRIED' equals to 1 if the admitted paitent is married.
    'MAR_SINGLE' equals to 1 if the admitted paitent is single.
    'MAR_UNKNOWN (DEFAULT)' equals to 1 if marital status of the admitted paitent is unknown.
    'MAR_WDS' equals to 1 if the admitted paitent is widowed, divorced, or seperated.
    'Readmission' equals to 1 if the patient was readmitted within 30 days.
    """
    feature_description_dict = {
        'LOS': "represents the total length of stay",
        'blood': "equals to 1 if the patient is diagnosed with blood and blood-forming organs diseases",
        'circulatory': "equals to 1 if the patient is diagnosed with circulatory system diseases",
        'congenital': "equals to 1 if the patient is diagnosed with congenital anomalies",
        'digestive': "equals to 1 if the patient is diagnosed with digestive system diseases",
        'endocrine': "equals to 1 if the patient is diagnosed with endocrine, nutritional and metabolic diseases, and immunity disorders",
        'genitourinary': "equals to 1 if the patient is diagnosed with henitourinary system diseases",
        'infectious': "equals to 1 if the patient is diagnosed with infectious and parasitic diseases",
        'injury': "equals to 1 if the patient is diagnosed with injury and poisoning",
        'mental': "equals to 1 if the patient is diagnosed with mental disorders",
        'muscular': "equals to 1 if the patient is diagnosed with musculoskeletal system and connective tissue diseases",
        'neoplasms': "equals to 1 if the patient is diagnosed with neoplasms",
        'nervous': "equals to 1 if the patient is diagnosed with nervous system and sense organs diseases",
        'respiratory': "equals to 1 if the patient is diagnosed with respiratory system diseases",
        'skin': "equals to 1 if the patient is diagnosed with skin and subcutaneous tissue diseases",
        'supp1': "equals to 1 if the patient satisfies supplementary classification of external causes of injury and poisoning",
        'supp2': "equals to 1 if the patient satisfies supplementary classification of factors influencing health status and contact with health services",
        'symptoms_signs': "equals to 1 if the patient shows symptoms, signs and ill-defined conditions",
        'GENDER': "equals to 1 if the patient is male",
        'age': "represents the age of the admitted patient",
        'ICU': "equals to 1 if the patient is admitted to the ICU",
        'ICU_LOS': "represents the total length of stay in ICU",
        'ADM_ELECTIVE': "equals to 1 if this admission is elective",
        'ADM_EMERGENCY': "equals to 1 if this admission is emergency",
        'ADM_URGENT': "equals to 1 if this admission is urgent",
        'INS_Government': "equals to 1 if the insurance of this admission is government",
        'INS_Medicaid': "equals to 1 if the insurance of this admission is Medicaid",
        'INS_Medicare': "equals to 1 if the insurance of this admission is Medicare",
        'INS_Private': "equals to 1 if the insurance of this admission is private",
        'INS_Self Pay': "equals to 1 if the insurance of this admission is self pay",
        'ETH_ASIAN': "equals to 1 if the admitted patient is Asian",
        'ETH_BLACK/AFRICAN AMERICAN': "equals to 1 if the admitted patient is Black/African American",
        'ETH_HISPANIC/LATINO': "equals to 1 if the admitted patient is Hispanic/Latino",
        'ETH_OTHER/UNKNOWN': "equals to 1 if ethnicity of the admitted patient is Other/Unknown",
        'ETH_WHITE': "equals to 1 if the admitted patient is White",
        'MAR_MARRIED': "equals to 1 if the admitted patient is married",
        'MAR_SINGLE': "equals to 1 if the admitted patient is single",
        'MAR_UNKNOWN (DEFAULT)': "equals to 1 if marital status of the admitted patient is unknown",
        'MAR_WDS': "equals to 1 if the admitted patient is widowed, divorced, or separated",
        'Readmission': "equals to 1 if the patient was readmitted within 30 days"
    }

    df = pd.read_csv("data/Testing_Data.csv",index_col=0)
    admission_feature = df.loc[df.index==admission_id,feature].astype(float).values[0]
    return f"The {feature} for admission {admission_id} is {admission_feature}. This feature {feature_description_dict}."

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
                              change: Annotated[float,"The amount of change for the specific feature column."])->str:
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
