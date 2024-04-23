# Python file of all tools that can be executed by the assistant
import pandas as pd
import joblib
from typing import Annotated

def get_admission_info(admission_id: Annotated[int, "The ID of the hospital admission to extract information for."], 
                     feature_lst: Annotated[list[str], "A list of feature columns to extract information for the admission."])->str:
    """
    Extracts information and provides descriptions for a list of feature columns for a hospital admission specified by its ID from a csv data file.
    """
    feature_description_dict = {
        'LOS': "represents the total length of stay",
        'blood': "equals to 1 if the patient is diagnosed with blood and blood-forming organs diseases and 0 otherwise",
        'circulatory': "equals to 1 if the patient is diagnosed with circulatory system diseases and 0 otherwise",
        'congenital': "equals to 1 if the patient is diagnosed with congenital anomalies and 0 otherwise",
        'digestive': "equals to 1 if the patient is diagnosed with digestive system diseases and 0 otherwise",
        'endocrine': "equals to 1 if the patient is diagnosed with endocrine, nutritional and metabolic diseases, and immunity disorders and 0 otherwise",
        'genitourinary': "equals to 1 if the patient is diagnosed with henitourinary system diseases and 0 otherwise",
        'infectious': "equals to 1 if the patient is diagnosed with infectious and parasitic diseases and 0 otherwise",
        'injury': "equals to 1 if the patient is diagnosed with injury and poisoning and 0 otherwise",
        'mental': "equals to 1 if the patient is diagnosed with mental disorders and 0 otherwise",
        'muscular': "equals to 1 if the patient is diagnosed with musculoskeletal system and connective tissue diseases and 0 otherwise",
        'neoplasms': "equals to 1 if the patient is diagnosed with neoplasms and 0 otherwise",
        'nervous': "equals to 1 if the patient is diagnosed with nervous system and sense organs diseases and 0 otherwise",
        'respiratory': "equals to 1 if the patient is diagnosed with respiratory system diseases and 0 otherwise",
        'skin': "equals to 1 if the patient is diagnosed with skin and subcutaneous tissue diseases and 0 otherwise",
        'supp1': "equals to 1 if the patient is diagnosed with supplementary classification of external causes of injury and poisoning and 0 otherwise",
        'supp2': "equals to 1 if the patient is diagnosed with supplementary classification of factors influencing health status and contact with health services and 0 otherwise",
        'symptoms_signs': "equals to 1 if the patient is diagnosed with symptoms, signs and ill-defined conditions and 0 otherwise",
        'GENDER': "equals to 1 if the patient is male and 0 otherwise",
        'age': "represents the age of the admitted patient",
        'ICU': "equals to 1 if the patient is admitted to the ICU and 0 otherwise",
        'ICU_LOS': "represents the total length of stay in ICU",
        'ADM_ELECTIVE': "equals to 1 if this admission is elective and 0 otherwise",
        'ADM_EMERGENCY': "equals to 1 if this admission is emergency and 0 otherwise",
        'ADM_URGENT': "equals to 1 if this admission is urgent and 0 otherwise",
        'INS_Government': "equals to 1 if the insurance of this admission is government and 0 otherwise",
        'INS_Medicaid': "equals to 1 if the insurance of this admission is Medicaid and 0 otherwise",
        'INS_Medicare': "equals to 1 if the insurance of this admission is Medicare and 0 otherwise",
        'INS_Private': "equals to 1 if the insurance of this admission is private and 0 otherwise",
        'INS_Self Pay': "equals to 1 if the insurance of this admission is self pay and 0 otherwise",
        'ETH_ASIAN': "equals to 1 if the admitted patient is Asian and 0 otherwise",
        'ETH_BLACK/AFRICAN AMERICAN': "equals to 1 if the admitted patient is Black/African American and 0 otherwise",
        'ETH_HISPANIC/LATINO': "equals to 1 if the admitted patient is Hispanic/Latino and 0 otherwise",
        'ETH_OTHER/UNKNOWN': "equals to 1 if ethnicity of the admitted patient is Other/Unknown and 0 otherwise",
        'ETH_WHITE': "equals to 1 if the admitted patient is White and 0 otherwise",
        'MAR_MARRIED': "equals to 1 if the admitted patient is married and 0 otherwise",
        'MAR_SINGLE': "equals to 1 if the admitted patient is single and 0 otherwise",
        'MAR_UNKNOWN (DEFAULT)': "equals to 1 if marital status of the admitted patient is unknown and 0 otherwise",
        'MAR_WDS': "equals to 1 if the admitted patient is widowed, divorced, or separated and 0 otherwise",
        'Readmission': "equals to 1 if the patient was readmitted within 30 days and 0 otherwise"
    }

    df = pd.read_csv("data/Testing_Data.csv",index_col=0)
    output_str = ""
    for feature in feature_lst:
        admission_feature = df.loc[df.index==admission_id,feature].astype(float).values[0]
        output_str+= f"The feature '{feature}' for admission {admission_id} is {admission_feature}. This feature {feature_description_dict[feature]}.\n"
    return output_str

def prepare_model_input(admission_id):
    """
    A helper function to prepare the data from an admission as input to the machine learning model for readmission prediction.
    This function is not registered by the assistant.
    """
    df = pd.read_csv("data/Testing_Data.csv",index_col=0)
    features = df.loc[df.index==admission_id,df.columns[:-1]].astype(float)
    return features

def make_readmission_prediction(admission_id: Annotated[int, "The ID of the hospital admission to predict readmission probability for."])->str:
    """
    Uses a trained machine learning model to predict the probability of readmission for a hospital admission specified by its ID based on all feature columns in the data.
    """
    admission_data = prepare_model_input(admission_id)
    X = admission_data.to_numpy()
    with open('data/lgb.pkl', 'rb') as f:
        model = joblib.load(f)
    predicted_prob = model.predict_proba(X,verbose=-1)[0][1]
    if predicted_prob > 0.5:
        return f"The machine learning model predicts a readmission probability of {'%.2f' % (predicted_prob)} for admission {admission_id}, so the patient will likely be readmitted within 30 days."
    else:
        return f"The machine learning model predicts a readmission probability of {'%.2f' % (predicted_prob)} for admission {admission_id}, so the patient will not likely be readmitted within 30 days."
    

def make_updated_readmission_prediction(admission_id: Annotated[int, "The ID of the hospital admission to predict the updated readmission probability for."],
                              updated_features: Annotated[dict[str, float], "A python dictionary that provides the updated feature values, with the feature names as keys and the updated feature values as values."])->str:
    """
    Uses a trained machine learning model to predict the updated probability of readmission for a hospital admission specified by its ID when some of its feature values change.
    The parameter 'updated_features' is a python dictionary that provides the updated feature values, with the feature names as keys and the updated feature values as values.
    For example, if 'updated_features' is '{'GENDER': 1}', it means the binary feature column 'GENDER' changes to 1 so the gender of the patient in this admission changes from female to male.
    """
    # Predict with original data
    admission_data = prepare_model_input(admission_id)
    X_old = admission_data.to_numpy()
    with open('data/lgb.pkl', 'rb') as f:
        model = joblib.load(f)
    predicted_prob_old = model.predict_proba(X_old,verbose=-1)[0][1]
    # Update feature values and predict with updated data
    admission_data_new = admission_data.copy()
    output = ""
    for name,new_val in updated_features.items():
        old_val = admission_data[name].values[0]
        admission_data_new[name]=new_val
        output += f"The feature '{name}' for admission {admission_id} changes from {old_val} to {new_val}.\n"
    X_new = admission_data_new.to_numpy()
    predicted_prob_new = model.predict_proba(X_new,verbose=-1)[0][1]
    if predicted_prob_new > 0.5:
        output += f"As a result, the predicted readmission probability changes from {'%.2f' % (predicted_prob_old)} to {'%.2f' % (predicted_prob_new)}, so the patient will likely be readmitted within 30 days."
    else:
        output += f"As a result, the predicted readmission probability changes from {'%.2f' % (predicted_prob_old)} to {'%.2f' % (predicted_prob_new)}, so the patient will not likely be readmitted within 30 days."
    return output
