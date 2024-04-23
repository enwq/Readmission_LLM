from openai import AzureOpenAI
import os

# Setup Azure OpenAI client   
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Define tools
tools = [{'type': 'function',
  'function': {
      'description': '''This tool extracts information and provides descriptions for a list of feature columns for a hospital admission specified by its ID from a csv data file.
      The required input parameters are 'admission_id' and 'feature_lst'.
      'admission_id' is an integer representing the ID of the hospital admission to extract information for.
      'feature_lst' is a list of strings representing feature columns to extract information for the admission.
      ''',    
   'name': 'get_admission_info',
   'parameters': {
      'type': 'object',
      'properties': {
        'admission_id': {
            'type': 'integer',
            'description': 'The ID of the hospital admission to extract information for.'
        },
        'feature_lst': {
            'type': 'array',
            'items': {
              'type':'string',
              'description':'Name of a feature column.'
            },
            'description': 'A list of feature columns to extract information for the admission.'
        }
      },
    'required': ['admission_id', 'feature_lst']}}},
    {'type': 'function',
      'function':{
        'description': '''This tool uses a trained machine learning model to predict the probability of readmission for a hospital admission specified by its ID based on all feature columns in the data.
        The required input parameter is 'admission_id'.
        'admission_id' is an integer representing the ID of the hospital admission to predict readmission probability for.
        ''',
        'name': 'make_readmission_prediction',
        'parameters':{
            'type':'object',
            'properties':{
                'admission_id':{
                    'type':'integer',
                    'description':'The ID of the hospital admission to predict readmission probability for.'
                }
            },
            'required':['admission_id']
        }
      }
    },
    {'type': 'function',
      'function':{
        'description': '''When some of the feature values of a hospital admission specified by its ID are updated, this tool uses a trained machine learning model to predict the updated probability of readmission.
        The required input parameters are 'admission_id' and 'updated_features'.
        'admission_id' is an integer representing the ID of the hospital admission to predict the updated readmission probability for.
        'updated_features' is a python dictionary that provides the updated feature values, with the feature names as keys and the updated feature values as values.
        For example, if 'updated_features' is '{'GENDER': 1}', it means the binary feature column 'GENDER' changes to 1 so the gender of the patient in this admission changes from female to male.
        ''',
        'name': 'make_updated_readmission_prediction',
        'parameters':{
            'type':'object',
            'properties':{
                'admission_id':{
                    'type':'integer',
                    'description':'The ID of the hospital admission to predict the updated readmission probability for.'
                },
                'updated_features':{
                    'type':'object',
                    'propertyNames':{
                        'type':'string',
                        'description':'Name of the feature to update value for.'
                    },
                    'additionalProperties':{
                        'type':'number',
                        'description':'Updated value for the feature.'
                    },
                    'description':'A python dictionary that provides the updated feature values, with the feature names as keys and the updated feature values as values.'
                }
            },
            'required':['admission_id','updated_features']
        }
      }
    },
    {'type': 'function',
      'function':{
        'description': '''This tool uses the SHAP (SHapley Additive exPlanations) algorithm to identify the 10 most important features used by a trained machine learning model for readmission prediction considering all available admission records from the data.
        It then generates a bar plot for the feature importance values of the identified features.
        This tool does not require any input parameter.
        ''',
        'name': 'compute_and_plot_shap_global_feature_importance',
        'parameters':{
            'type':'object',
            'properties':{},
            'required':[]
        }
      }
    }

 ]

# Create assistant
assistant = client.beta.assistants.create(
  name="Medical Assistant",
  instructions='''You are a helpful AI medcical assistant for doctors to answer their questions about patient admissions.
  Hospital admission data is provided in a csv file with the following feature columns.

  'LOS' is the total length of stay.
  'blood' equals to 1 if the patient is diagnosed with blood and blood-forming organs diseases and 0 otherwise.
  'circulatory' equals to 1 if the patient is diagnosed with circulatory system diseases and 0 otherwise.
  'congenital' equals to 1 if the patient is diagnosed with congenital anomalies and 0 otherwise.
  'digestive' equals to 1 if the patient is diagnosed with digestive system diseases and 0 otherwise.
  'endocrine' equals to 1 if the patient is diagnosed with endocrine, nutritional and metabolic diseases, and immunity disorders and 0 otherwise.
  'genitourinary' equals to 1 if the patient is diagnosed with henitourinary system diseases and 0 otherwise.
  'infectious' equals to 1 if the patient is diagnosed with infectious and parasitic diseases and 0 otherwise.
  'injury' equals to 1 if the patient is diagnosed with injury and poisoning and 0 otherwise.
  'mental' equals to 1 if the patient is diagnosed with mental disorders and 0 otherwise.
  'muscular' equals to 1 if the patient is diagnosed with musculoskeletal system and connective tissue diseases and 0 otherwise.
  'neoplasms' equals to 1 if the patient is diagnosed with neoplasms and 0 otherwise.
  'nervous' equals to 1 if the patient is diagnosed with nervous system and sense organs diseases and 0 otherwise.
  'respiratory' equals to 1 if the patient is diagnosed with respiratory system diseases and 0 otherwise.
  'skin' equals to 1 if the patient is diagnosed with skin and subcutaneous tissue diseases and 0 otherwise.
  'supp1' equals to 1 if the patient is diagnosed with supplementary classification of external causes of injury and poisoning and 0 otherwise.
  'supp2' equals to 1 if the patient is diagnosed with supplementary classification of factors influencing health status and contact with health services and 0 otherwise.
  'symptoms_signs' equals to 1 if the patient is diagnosed with symptoms, signs and ill-defined conditions and 0 otherwise.
  'GENDER' equals to 1 if the patient is male and 0 otherwise.
  'age' is the age of the admitted patient.
  'ICU' equals to 1 if the patient is admitted to the ICU and 0 otherwise.
  'ICU_LOS' is the total length of stay in ICU.
  'ADM_ELECTIVE' equals to 1 if this admission is elective and 0 otherwise.
  'ADM_EMERGENCY' equals to 1 if this admission is emergency and 0 otherwise.
  'ADM_URGENT' equals to 1 if this admission is urgent and 0 otherwise.
  'INS_Government' equals to 1 if the insurance of this admission is government and 0 otherwise.
  'INS_Medicaid' equals to 1 if the insurance of this admission is Medicaid and 0 otherwise.
  'INS_Medicare' equals to 1 if the insurance of this admission is Medicare and 0 otherwise.
  'INS_Private' equals to 1 if the insurance of this admission is private and 0 otherwise.
  'INS_Self Pay' equals to 1 if the insurance of this admission is self pay and 0 otherwise.
  'ETH_ASIAN' equals to 1 if the admitted patient is Asian and 0 otherwise.
  'ETH_BLACK/AFRICAN AMERICAN' equals to 1 if the admitted patient is Black/African American and 0 otherwise.
  'ETH_HISPANIC/LATINO' equals to 1 if the admitted patient is Hispanic/Latino and 0 otherwise.
  'ETH_OTHER/UNKNOWN' equals to 1 if ethinicity of the admitted patient is Other/Unknown and 0 otherwise.
  'ETH_WHITE' equals to 1 if the admitted patient is White and 0 otherwise.
  'MAR_MARRIED' equals to 1 if the admitted paitent is married and 0 otherwise.
  'MAR_SINGLE' equals to 1 if the admitted paitent is single and 0 otherwise.
  'MAR_UNKNOWN (DEFAULT)' equals to 1 if marital status of the admitted paitent is unknown and 0 otherwise.
  'MAR_WDS' equals to 1 if the admitted paitent is widowed, divorced, or seperated and 0 otherwise.
  'Readmission' equals to 1 if the patient was readmitted within 30 days and 0 otherwise.

  You are provided with a list of tools and you need to select the most appropriate tool to answer doctors' questions.
  You must specify all required parameters for your selected tool.
  Do not make assumptions about what parameter values to use for these tools. Ask for clarification if the user's request is ambiguous.
  If none of the tools is suitable to answer the given question, answer with your own knowledge.
  ''',
  tools=tools,
  model="MIMIC",
)

# Print out the assistant
print(assistant.model_dump_json(indent=2))