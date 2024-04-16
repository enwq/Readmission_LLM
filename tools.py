# Python file of all tools that can be executed by the agents
import pandas as pd
import os

def get_subject_info(file_name: str, subject: str, feature: str)->str:
    """
    Use this function to extract information of a particular feature column for a specific subject from a csv file.
    file_name is the name of the csv file that contains all data.
    subject is the name of the subject to extract information for.
    feature is the name of the column in the data to extract information for.
    """
    df = pd.read_csv(os.path.join("data",file_name))
    subject_line = df.loc[df['Name']==subject]
    subject_feature = subject_line[feature].values[0]
    return f"The {feature} for {subject} is {subject_feature}."

