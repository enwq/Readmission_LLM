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
  'function': {'description': """
    Extracts information and provides descriptions for a list of feature columns for a hospital admission specified by its ID from a csv data file.
    """,
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
              'type':'string'
            },
            'description': 'A list of feature columns to extract information for the admission.'
        }
      },
    'required': ['admission_id', 'feature_lst']}}}
 ]

# Create assistant
assistant = client.beta.assistants.create(
  name="Assistant",
  instructions='''
  You are a helpful AI assistant for doctors to answer their questions about patient admissions.
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
  If none of the tools is suitable to answer the given question, answer with your own knowledge.
  ''',
  tools=tools,
  model="MIMIC",
)

# Print out the assistant
print(assistant.model_dump_json(indent=2))