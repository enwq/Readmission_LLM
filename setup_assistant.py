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
  'function': {'description': 'A tool to extract information of a particular feature column for a specific subject from a csv file.',
   'name': 'get_subject_info',
   'parameters': {'type': 'object',
    'properties': {'file_name': {'type': 'string',
      'description': 'The name of the csv file that contains all data.'},
     'subject': {'type': 'string',
      'description': 'The name of the subject to extract information for.'},
     'feature': {'type': 'string',
      'description': 'The name of the column in the data to extract information for.'}},
    'required': ['file_name', 'subject', 'feature']}}},
 {'type': 'function',
  'function': {'description': 'A tool to predict the price of the given subject using a trained machine learning model.',
   'name': 'make_price_prediction',
   'parameters': {'type': 'object',
    'properties': {'file_name': {'type': 'string',
      'description': 'The name of the csv file that contains all data.'},
     'subject': {'type': 'string',
      'description': 'The name of the subject to predict price for.'}},
    'required': ['file_name', 'subject']}}},
 {'type': 'function',
  'function': {'description': 'A tool to compute the change in the predicted price of the given subject using a trained machine learning model, when the value of a speific feature is changed by a certain amount.',
   'name': 'compute_prediction_change',
   'parameters': {'type': 'object',
    'properties': {'file_name': {'type': 'string',
      'description': 'The name of the csv file that contains all data.'},
     'subject': {'type': 'string',
      'description': 'The name of the subject to predict price for.'},
     'feature': {'type': 'string',
      'description': 'The name of the column in the data that changes.'},
     'change': {'type': 'number',
      'description': 'The amount of change for the specific feature column.'}},
    'required': ['file_name', 'subject', 'feature', 'change']}}}]

# Create assistant
assistant = client.beta.assistants.create(
  name="Assistant",
  instructions='''
  You are a helpful AI assistant to select the most appropriate tool to answer the user's question.
  If none of the tools is suitable to answer the given question, answer with your own knowledge.
  ''',
  tools=tools,
  model="MIMIC",
)

# Print out the assistant
print(assistant.model_dump_json(indent=2))