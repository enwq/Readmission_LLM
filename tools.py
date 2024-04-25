# Python file of all tools that can be executed by the assistant
import pandas as pd
import joblib
from typing import Annotated
import shap
import matplotlib.pyplot as plt
import numpy as np
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
import os

def get_admission_info(admission_id: Annotated[int, "The ID of the hospital admission to extract information for."], 
                     feature_lst: Annotated[list[str], "A list of feature columns to extract information for the admission."])->str:
    """
    This tool extracts information and provides descriptions for a list of feature columns for a hospital admission specified by its ID from a csv data file.
    The required input parameters are 'admission_id' and 'feature_lst'.
    'admission_id' is an integer representing the ID of the hospital admission to extract information for.
    'feature_lst' is a list of strings representing feature columns to extract information for the admission.
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

def prepare_risk_input(admission_id):
    """
    A helper function to prepare the data from an admission as input to the risk score model for readmission prediction.
    This function is not registered by the assistant.
    """
    df = pd.read_csv("data/Testing_Data_Discretized.csv",index_col=0)
    features_df = df.drop(columns=['Readmission'])
    features = features_df.loc[features_df.index==admission_id].astype(float)
    return features

def make_readmission_prediction(admission_id: Annotated[int, "The ID of the hospital admission to predict readmission probability for."])->str:
    """
   This tool uses a trained machine learning model as well as a risk score model to predict the probability of readmission for a hospital admission specified by its ID.
   The required input parameter is 'admission_id'.
   'admission_id' is an integer representing the ID of the hospital admission to predict readmission probability for.
    """
    # Prediction with ML model
    admission_data = prepare_model_input(admission_id)
    X = admission_data.to_numpy()
    with open('data/lgb.pkl', 'rb') as f:
        model = joblib.load(f)
    predicted_prob = model.predict_proba(X,verbose=-1)[0][1]*100
    ml_output = f"The machine learning model predicts a readmission probability of {'%.1f' % (predicted_prob)}% for admission {admission_id}.\n"
    # Prediction with risk score model
    admission_data_risk = prepare_risk_input(admission_id)
    X_risk = admission_data_risk.to_numpy()
    with open('data/faster_risk.pkl', 'rb') as f:
        risk_model = joblib.load(f)
    predicted_risk = risk_model.predict_prob(X_risk)[0]*100
    risk_output = f"The risk score model predicts a readmission risk of {'%.1f' % (predicted_risk)}% for admission {admission_id}."
    return ml_output+risk_output
    
def handle_mutual_exclusive_feature_update(feature_name):
    '''
    Handles feature update for mutually exclusive binary columns, i.e. admission type, insurance type, ethnicity, and marital status.
    feature_name is the column to be updated as 1.
    This function is not registered by the assistant.
    '''
    adm_features = ['ADM_ELECTIVE','ADM_EMERGENCY','ADM_URGENT']
    ins_features = ['INS_Government', 'INS_Medicaid', 'INS_Medicare', 'INS_Private', 'INS_Self Pay']
    eth_features = ['ETH_ASIAN', 'ETH_BLACK/AFRICAN AMERICAN', 'ETH_HISPANIC/LATINO', 'ETH_OTHER/UNKNOWN', 'ETH_WHITE']
    mar_features = ['MAR_MARRIED', 'MAR_SINGLE','MAR_UNKNOWN (DEFAULT)', 'MAR_WDS']
    # Return the feature names to be updated as 0 to maintain mutual exclusion
    if feature_name.startswith('ADM'):
        return [f for f in adm_features if f!=feature_name]
    elif feature_name.startswith('INS'):
        return [f for f in ins_features if f!=feature_name]
    elif feature_name.startswith('ETH'):
        return [f for f in eth_features if f!=feature_name]
    else:
        return [f for f in mar_features if f!=feature_name]
    
def handle_discretized_feature_update(feature_name,feature_value):
    '''
    Handles feature update for continuous features, i.e. LOS, age, ICU_LOS.
    Adjust their corresponding discretized feature columns correspondingly.
    Returns the names of columns to update and the updated values.
    '''
    # Load the fitted discretizer and corresponding feature names
    with open(f'data/{feature_name}_discretizer.pkl', 'rb') as f:
        discretizer = joblib.load(f)
    with open(f'data/{feature_name}_discretized_names.pkl', 'rb') as f:
        discretized_names = joblib.load(f)
    # Discretize the updated feature value
    new_values = discretizer.transform(np.array(feature_value).reshape((-1,1)))[0]
    return new_values,discretized_names

def make_updated_readmission_prediction(admission_id: Annotated[int, "The ID of the hospital admission to predict the updated readmission probability for."],
                              updated_features: Annotated[dict[str, float], "A python dictionary that provides the updated feature values, with the feature names as keys and the updated feature values as values."])->str:
    """
    Use this tool to answer any question related to how the predicted probability of readmission according to the trained machine learning model as well as the risk score model would be updated for an admission specified by its ID if some features values of this admission are updated.
    The required input parameters are 'admission_id' and 'updated_features'.
    'admission_id' is an integer representing the ID of the hospital admission to predict the updated readmission probability for.
    'updated_features' is a python dictionary that provides the updated feature values, with the feature names as keys and the updated feature values as values.
    An example usage of this tool is provided below.

    Question: For the patient with admission ID 53631, how will the predicted probability of readmission change if this patient is a female and stays in hospital for 28 days?
    Tool call: make_updated_readmission_prediction('admission_id':53631,'updated_features':{'GENDER':0.0,'LOS':28.0})
    This tool then uses the trained machine learning model and the risk score model to predict readmission probability for the admission record with ID 53631 where the feature values for 'GENDER' and 'LOS' are updated to be 0.0 and 28.0 respectively.
    """
    # ML model prediction
    # Predict with original data
    admission_data = prepare_model_input(admission_id)
    X_old = admission_data.to_numpy()
    with open('data/lgb.pkl', 'rb') as f:
        model = joblib.load(f)
    predicted_prob_old = model.predict_proba(X_old,verbose=-1)[0][1]*100
    # Update feature values and predict with updated data
    admission_data_new = admission_data.copy()
    ml_output = ""
    for name,new_val in updated_features.items():
        old_val = admission_data[name].values[0]
        admission_data_new[name]=new_val
        # Need to uniform the updates for ICU and ICU_LOS
        if name=="ICU" and new_val==0.0:
            admission_data_new["ICU_LOS"]=0.0
        if name=="ICU_LOS" and new_val==0.0:
            admission_data_new["ICU"]=0.0
        if name=="ICU_LOS" and new_val>0.0:
            admission_data_new["ICU"]=1.0
        # Adjust correspondingly for mutually exclusive columns
        if (name.startswith('ADM') or name.startswith('INS') or name.startswith('ETH') or name.startswith('MAR')) and new_val==1.0:
            columns_to_adjust = handle_mutual_exclusive_feature_update(name)
            admission_data_new[columns_to_adjust]=0.0
        ml_output += f"The feature '{name}' for admission {admission_id} changes from {old_val} to {new_val}.\n"
    X_new = admission_data_new.to_numpy()
    predicted_prob_new = model.predict_proba(X_new,verbose=-1)[0][1]*100
    ml_output += f"As a result, the predicted readmission probability from the machine learning model changes from {'%.1f' % (predicted_prob_old)}% to {'%.1f' % (predicted_prob_new)}%.\n"
    # Risk score model prediction
    # Predict with original data
    admission_data_risk = prepare_risk_input(admission_id)
    X_risk_old = admission_data_risk.to_numpy()
    with open('data/faster_risk.pkl', 'rb') as f:
        risk_model = joblib.load(f)
    predicted_risk_old = risk_model.predict_prob(X_risk_old)[0]*100
    # Update feature values and predict with updated data
    admission_data_risk_new = admission_data_risk.copy()
    for name,new_val in updated_features.items():
        # Adjust continuous features
        if name in ['LOS','age','ICU_LOS']:
            discretized_values,discretized_names = handle_discretized_feature_update(name,new_val)
            for v,n in zip(discretized_values,discretized_names):
                admission_data_risk_new[n]=v
            # Need to uniform the updates for ICU and ICU_LOS
            if name=="ICU_LOS" and new_val==0.0:
                admission_data_risk_new["ICU"]=0.0
            if name=="ICU_LOS" and new_val>0.0:
                admission_data_risk_new["ICU"]=1.0
        # Adjust binary features
        else:
            # Need to uniform the updates for ICU and ICU_LOS
            if name=="ICU" and new_val==0.0:
                discretized_values,discretized_names = handle_discretized_feature_update("ICU_LOS",0.0)
                for v,n in zip(discretized_values,discretized_names):
                    admission_data_risk_new[n]=v
            # Adjust correspondingly for mutually exclusive columns
            admission_data_risk_new[name]=new_val
            # Adjust correspondingly for mutually exclusive columns
            if (name.startswith('ADM') or name.startswith('INS') or name.startswith('ETH') or name.startswith('MAR')) and new_val==1.0:
                columns_to_adjust = handle_mutual_exclusive_feature_update(name)
                admission_data_risk_new[columns_to_adjust]=0.0
    risk_output = ""
    X_risk_new = admission_data_risk_new.to_numpy()
    predicted_risk_new = risk_model.predict_prob(X_risk_new)[0]*100
    risk_output += f"The predicted readmission probability from the risk score model changes from {'%.1f' % (predicted_risk_old)}% to {'%.1f' % (predicted_risk_new)}%."
    return ml_output+risk_output

def compute_shap_values():
    """
    A helper function to compute shap values for all testing records.
    This function is not registered by the assistant.
    """
    # Load the data
    df = pd.read_csv("data/Testing_Data.csv",index_col=0)
    feature_names = df.columns[:-1]
    X = df[feature_names].astype(float).to_numpy()
    # Load the model
    with open('data/lgb.pkl', 'rb') as f:
        model = joblib.load(f)
    # Get SHAP explainer
    explainer = shap.Explainer(model)
    # Compute SHAP values
    shap_values = explainer(X)
    shap_values.feature_names = feature_names
    return shap_values

def compute_and_plot_shap_global_feature_importance()->str:
    """
    This tool uses the SHAP (SHapley Additive exPlanations) algorithm to identify the 10 most important features used by a trained machine learning model for readmission prediction considering all available admission records from the data.
    It then generates a bar plot for the feature importance values of the identified features.
    This tool does not require any input parameter.
    """
    # Compute SHAP values
    shap_values = compute_shap_values()
    # Save the SHAP values in a dataframe
    shap_mean_abs = np.abs(shap_values.values).mean(0)
    shap_values_df = pd.DataFrame(shap_mean_abs,columns=["mean(|SHAP value|)"])
    shap_values_df['feature'] = shap_values.feature_names
    shap_values_df.sort_values('mean(|SHAP value|)',ascending=False,inplace=True,ignore_index=True)
    # Get the top 9 most important features + combined shap values from the remaining features
    top_9_feature_names = shap_values_df['feature'].values[:9]
    top_9_shap_values = shap_values_df['mean(|SHAP value|)'].values[:9]
    formatted_values = [float('%.3f' % (num)) for num in top_9_shap_values]
    output = f"According to the SHAP (SHapley Additive exPlanations) algorithm, the top 9 most important features for readmission prediction are {top_9_feature_names}, with effects of {formatted_values} respectively.\n"
    remaining_shap_sum = np.sum(shap_values_df['mean(|SHAP value|)'].values[9:])
    output += f"The remaining {len(shap_values.feature_names)-9} features show a combined effect of {'%.3f' % (remaining_shap_sum)} for readmission prediction."
    # Generate and save the bar plot
    shap.plots.bar(shap_values, max_display=10,show=False)
    plt.title('Top 10 most important features')
    plt.savefig('plot/global_top_10_features.png', bbox_inches = 'tight')
    return output

def compute_and_plot_shap_local_feature_importance(admission_id: Annotated[int, "The ID of the hospital admission to compute SHAP feature importance for."])->str:
    """
    This tool uses the SHAP (SHapley Additive exPlanations) algorithm to identify the top 10 features that contribute the most to the trained machine learning model for predicting readmission for the admission specified by its ID.
    It then generates a waterfall plot to show the effects of the identified features on predicting readmission for the admission specified by its ID.
    The required input parameters is 'admission_id'.
    'admission_id' is an integer representing the ID of the hospital admission to compute SHAP feature importance for.
    """
    # Load data to get position of admission_id in the data
    df = pd.read_csv("data/Testing_Data.csv",index_col=0)
    idx = df.index.get_loc(admission_id)
    # Compute SHAP values
    shap_values = compute_shap_values()
    # Get SHAP values for the admission
    admission_shape_values = shap_values[idx]
    # Save the shape values in a dataframe
    admission_shap_values_df = pd.DataFrame(admission_shape_values.values,columns=["SHAP value"])
    admission_shap_values_df['|SHAP value|'] = np.abs(admission_shape_values.values)
    admission_shap_values_df['feature'] = admission_shape_values.feature_names
    admission_shap_values_df['data'] = admission_shape_values.data
    admission_shap_values_df.sort_values('|SHAP value|',ascending=False,inplace=True,ignore_index=True)
    # Get the top 9 most important features + combined shap values from the remaining features
    top_9_feature_names = admission_shap_values_df['feature'].values[:9]
    top_9_shap_values = admission_shap_values_df['SHAP value'].values[:9]
    top_9_feature_values = admission_shap_values_df['data'].values[:9]
    formatted_shap_values = [float('%.3f' % (num)) for num in top_9_shap_values]
    formatted_feature_values = [float('%.3f' % (num)) for num in top_9_feature_values]
    output = f"According to the SHAP (SHapley Additive exPlanations) algorithm, the top 9 features that contribute the most to readmission prediction for admission {admission_id} are {top_9_feature_names}, with effects of {formatted_shap_values} respectively.\n"
    output += f"The values of the top 9 features are {formatted_feature_values} respectively.\n"
    remaining_shap_sum = np.sum(admission_shap_values_df['SHAP value'].values[9:])
    output += f"The remaining {len(admission_shape_values.feature_names)-9} features show a combined effect of {'%.3f' % (remaining_shap_sum)} for readmission prediction."
    # Generate and save the waterfall plot
    shap.plots.waterfall(admission_shape_values, max_display=10,show=False)
    plt.title(f'Top 10 most important features for admission {admission_id}')
    plt.savefig('plot/local_top_10_features.png', bbox_inches = 'tight')
    return output

def get_risk_score_model_information():
    '''
    This tool prints out the detailed information about the risk score model used for readmission prediction, including features used, scoring of each feature, and corresponding risk of readmission at each score level.
    This tool does not require any input parameter.
    '''
    with open('data/risk_score_model_card.txt', 'r') as file:
        model_card = file.read()
    return model_card

def explain_risk_score_algorithm(question: Annotated[str, "The question about the risk score algorithm to answer."])->str:
    """
    Use this tool to answer any question related to the FasterRisk algorithm used by the risk score model. Ths tool extracts information from the academic paper about this algorithm to answer the question.
    The required input parameters is 'question'.
    'question' is a string representing the question about the risk score algorithm to answer.
    """
    # Load embedding database for the paper
    embeddings = AzureOpenAIEmbeddings(
            azure_deployment="MIMIC_text_embedding",
            openai_api_version="2024-02-15-preview",
        )
    faiss_index = FAISS.load_local(f'data/FasterRisk_faiss_db', embeddings,allow_dangerous_deserialization=True)
    
    docs = faiss_index.similarity_search(question)

    # Get the best matched page
    full_text = docs[0].page_content

    # Summarize the information
    prompt = f"""Using the following information below, you are asked to answer the user's question.

    {full_text}

    Answer:"""
    
    client = client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    response = client.chat.completions.create(
        model="MIMIC",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"{question}"}
        ]
    )
    answer = response.choices[0].message.content
    return answer
