from dotenv import load_dotenv
import os
from langchain.embeddings import SentenceTransformerEmbeddings
import glob
from langchain.vectorstores import FAISS
import textwrap
import random
import numpy as np
import json
from typing import List, Tuple
import sys
import time
from colorama import Fore, Style, init
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from huggingface_hub import login

load_dotenv()
login(token = os.environ.get("HF_TOKEN"))
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

def print_typing_effect(text: str, delay: float = 0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()
def format_message(role: str, content: str) -> str:
    return f"{role.capitalize()}: {content}"
def get_chat_history(memory) -> List[Tuple[str, str]]:
    return [(msg.type, msg.content) for msg in memory.chat_memory.messages]
def retrieve_from_faiss(query, loc, embeddings, top_k=2):
    vector_store = FAISS.load_local(loc, embeddings, allow_dangerous_deserialization=True)
    
    query_embedding = embeddings.embed_query(query['input'])
    
    # Retrieve the top 2 results
    results = vector_store.similarity_search_by_vector(query_embedding, top_k)
    combined_text = " ".join([result.page_content for result in results])
    
    return combined_text

def contains_uncertainty(response):
    uncertain_phrase = "I need to check"
    if uncertain_phrase in response.content:
        return True
    return False
def get_confidence(response):
    avg_confidence = 0
    for i in range(len(response.response_metadata['logprobs']['content'])):
        avg_confidence = avg_confidence + response.response_metadata['logprobs']['content'][i]['logprob']
    avg_confidence = avg_confidence / len(response.response_metadata['logprobs']['content'])

    return avg_confidence

def is_response_confident(response):
    # 1. Check if response contains uncertain language
    if contains_uncertainty(response):
        return False

    confidence_score = get_confidence(response)
    # 2. Check log probability confidence (if available)
    if confidence_score < -1.5:
        return False

    return True

def get_answer(chain, query, embed_model):
    initial_response = chain.invoke(query)

    if is_response_confident(initial_response):
        return initial_response

    loc = "./faq_db"
    relevant_docs = retrieve_from_faiss(query, loc, embed_model, top_k=5)

    # if not relevant_docs:
    #     return "I couldn't find additional information."

    augmented_query = {'input': f"Context:\n" + "\n".join(relevant_docs) + f"\n\nQuestion: {query['input']}"}

    final_response = chain.invoke(augmented_query)

    return final_response

def run_chatgpt_chatbot(llm_id, query, provider):
    if provider=="OpenAI":
        model=ChatOpenAI(model_name=llm_id, temperature=0.7, logprobs=True)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    SYS_PROMPT = """Answer the following question elaborately only if you're certain. Otherwise, say 'I need to check': {input}"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', SYS_PROMPT),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}')
        ]
    )
    memory = ConversationBufferWindowMemory(k=30, return_messages=True)
    conversation_chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
        )
    
        | prompt
    
        | model
    )

    user_input = query
    # if user_input.strip().upper() == 'STOP':
    #     print_typing_effect('ChatGPT: Goodbye! It was a pleasure chatting with you.')
    #     break
    # elif user_input.strip().upper() == 'HISTORY':
    #     chat_history = get_chat_history(memory)
    #     print("\n--- Chat History ---")
    #     for role, content in chat_history:
    #         print(format_message(role, content))
    #     print("--- End of History ---\n")
    #     continue
    # elif user_input.strip().upper() == 'CLEAR':
    #     memory.clear()
    #     print_typing_effect('ChatGPT: Chat history has been cleared.')
    #     continue

    user_inp = {'input': user_input}
    reply = get_answer(chain = conversation_chain, query = user_inp, embed_model = embeddings)
    memory.save_context(user_inp, {'output': reply.content})

    return reply.content
