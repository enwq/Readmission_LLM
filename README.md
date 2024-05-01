# Integrated AI Framework for Hospital Readmission Prediction and Clinical Decision Support
Nguyen Le, Ted Sohn, Yufei Shen

The University of Texas at Austin, Austin, Texas, USA

## Get Started
1. Download `ADMISSIONS.csv`, `PATIENTS.csv`, `ICUSTAYS.csv`, and `DIAGNOSES_ICD.csv` from MIMIC-III database. Set their appropriate paths in `HospitalReadmission.ipynb`.
2. Run `HospitalReadmission.ipynb` to train and evaluate the machine learning model for readmission prediction.
3. Move the output files (`Training_Data.csv`, `Testing_Data.csv`, `prediction_model.pkl`, and `hyperparameter_tuning.pkl`) to the data folder.
4. Run `readmission_fasterrisk.ipynb` to train and evaluate the risk score model using the FasterRisk algorithm for readmission prediction.
5. Copy and paste the model information to `data/risk_score_model_card.txt`.
6. Configure Azure OpenAI api key and end point. Deploy `GPT-3.5-Turbo (version 0613)` with name `MIMIC` and `text-embedding-3-small` with name `MIMIC_text_embedding` on Azure OpenAI studio.
6. Run `document_search.ipynb` to obtain the text embeddings for retrieval-augmented generation on the FasterRisk paper.
7. Call `setup_assistant.py` to set up the LLM-based medical assistant using OpenAI Assistant API.
8. Copy and paste the ID (`asst_XXXXXX`) of the assistant to assistant_chatbot.py and call `streamlit run assistant_chatbot.py` on command line to launch the assistant.
