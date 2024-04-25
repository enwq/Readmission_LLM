import streamlit as st
from openai import AzureOpenAI
import os
from tools import get_admission_info
from tools import make_readmission_prediction,make_updated_readmission_prediction
from tools import compute_and_plot_shap_global_feature_importance,compute_and_plot_shap_local_feature_importance
from tools import get_risk_score_model_information
import json

st.set_page_config(layout="centered")
@st.cache_resource
def setup():
    # Initialize the client
    client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    assistant = client.beta.assistants.retrieve("asst_SGGRG974voQhNDRuTVA5DcSa")
    thread = client.beta.threads.create()
    return client,assistant,thread

client,assistant,thread = setup()

function_dispatch_table = {
        "get_admission_info": get_admission_info,
        "make_readmission_prediction": make_readmission_prediction,
        "make_updated_readmission_prediction": make_updated_readmission_prediction,
        "compute_and_plot_shap_global_feature_importance":compute_and_plot_shap_global_feature_importance,
        "compute_and_plot_shap_local_feature_importance":compute_and_plot_shap_local_feature_importance,
        "get_risk_score_model_information":get_risk_score_model_information
}

def get_response(user_input):
     
    # Create the message
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )
    
    # Start running
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Define the list to store tool outputs
    tool_outputs = []

    # Define the image to display
    image_to_display = None

    # Loop through each tool in the required action section
    if run.status == 'requires_action':
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            tool_name = tool.function.name
            tool_args = json.loads(tool.function.arguments)
            print(tool_name)
            print(tool_args)
            # Execute the corresponding function and add the returned results
            func = function_dispatch_table.get(tool_name)
            if func:
                result = func(**tool_args)
                tool_outputs.append({"tool_call_id": tool.id, "output": result})
                # Get the local and global feature importance plot
                if tool_name == "compute_and_plot_shap_global_feature_importance":
                    image_to_display = 'plot/global_top_10_features.png'
                if tool_name == "compute_and_plot_shap_local_feature_importance":
                    image_to_display = 'plot/local_top_10_features.png'
            else:
                print(f"Function {tool_name} not found.")
        
        # Submit all tool outputs at once after collecting them in a list
        if tool_outputs:
            try:
                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
                )
            except Exception as e:
                print("Failed to submit tool outputs:", e)

    # Get the response messages
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)  
        output = messages.data[0].content[0].text.value
    else:
        output = "Failed to generate response."
    return output,image_to_display

# Default to empty text for user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Action when user enters text
def submit():
   st.session_state.user_input = st.session_state.query
   st.session_state.query = ""

st.title("Chatbot")

# Text input box 
st.text_input("Enter text:",key="query",on_change=submit)

# Displays user input
user_input = st.session_state.user_input
st.write("Your input:",user_input)

# Generate response for user input
if user_input:
    result,image_to_display = get_response(user_input)
    st.header("Assistant")
    st.write(result)
    if image_to_display:
        st.image(image_to_display)
