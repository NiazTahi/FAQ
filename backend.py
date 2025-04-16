# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from FAQ import run_chatgpt_chatbot

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    messages: str


#Step2: Setup AI Agent from FrontEnd Request


ALLOWED_MODEL_NAMES=["gpt-3.5-turbo-instruct", "gpt-4-turbo"]

app=FastAPI(title="FAQ SocioFi")

@app.post("/chat")
def chat_endpoint(request: RequestState): 
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}
    
    llm_id = request.model_name
    query = request.messages
    provider = request.model_provider

    # Create AI Agent and get response from it! 
    response=run_chatgpt_chatbot(llm_id, query, provider)
    return response

#Step3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
