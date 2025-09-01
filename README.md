# Summarizer for Private Legal Documents


## File Structure
- LegalSummarizer: Main Repository 
    |-> .gitignore
    |-> app.py : Contains the main logic for the LegalSummarizer application. It exposes an API endpoint that allows the user of the API to call this service like any other REST API.
    |-> requirements.txt
    |-> Procfile: Provides the start command for Google Run to run this application.
    |-> runtime.txt
    |-> **text.py : Test code to call the service via Google Cloud Platform.**

## Required Environment Variables
- HF_TOKEN: A huggingface access token to access the model used to create a privacy filter (google/flan-t5-small)
- GOOGLE_APPLICATION_CREDENTIALS: Must contain the contents of the service account json file or must be able to access the json file.













<!-- ---
title: Legal Document Summarizer
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.29.0"
app_file: app.py
pinned: false
--- -->