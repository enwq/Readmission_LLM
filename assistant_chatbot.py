import streamlit as st
from openai import AzureOpenAI
import os
from tools import get_admission_info,make_price_prediction,compute_prediction_change
import json

st.set_page_config(layout="wide")
@st.cache_resource
def setup():
    # Initialize the client
    client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    assistant = client.beta.assistants.retrieve("asst_54Gag3VQviHVRe7X6ad6LlGs")
    thread = client.beta.threads.create()
    return client,assistant,thread

client,assistant,thread = setup()

function_dispatch_table = {
        "get_admission_info": get_admission_info,
        "make_price_prediction": make_price_prediction,
        "compute_prediction_change": compute_prediction_change
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
    
    # Loop through each tool in the required action section
    if run.status == 'requires_action':
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            tool_name = tool.function.name
            tool_args = json.loads(tool.function.arguments)
            # Execute the corresponding function and add the returned results
            func = function_dispatch_table.get(tool_name)
            if func:
                result = func(**tool_args)
                tool_outputs.append({"tool_call_id": tool.id, "output": result})
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
    return output

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
    result = get_response(user_input)
    st.header("Assistant")
    st.text(result)
