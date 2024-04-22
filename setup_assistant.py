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
    Extracts information and provides desciptions for a list of feature columns for a hospital admission specified by its ID from a csv data file.
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
    'supp1' equals to 1 if the patient is diagnosed with supplementary classification of external causes of injury and poisoning.
    'supp2' equals to 1 if the patient is diagnosed with supplementary classification of factors influencing health status and contact with health services.
    'symptoms_signs' equals to 1 if the patient shows symptoms, signs and ill-defined conditions.
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
  You are a helpful AI assistant for doctors. 
  You are provided with a list of tools and you need to select the most appropriate tool to answer doctors' questions.
  If none of the tools is suitable to answer the given question, answer with your own knowledge.
  ''',
  tools=tools,
  model="MIMIC",
)

# Print out the assistant
print(assistant.model_dump_json(indent=2))