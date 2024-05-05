from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from modules.WebSearcher import WebSearcher
from modules.Agent import AgentWithSearchTool

from modules.RedisProvider import Cache
from upstash_redis import Redis


import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class Query(BaseModel):
    prompt: str

app = FastAPI(
    title='Cheko App ✔' 
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0.8)
web_searcher = WebSearcher()

redis_client = Cache(host=os.getenv('UPSTASH_REDIS_HOST'), password=os.getenv('UPSTASH_REDIS_PASSWORD'))
if redis_client:
    print('Redis connected ✔')
else:
    print('Redis not connected ❌')

@app.get('/ping')
def ping_pong():
    return {'message': 'Pong ✔'}

@app.post('/api/v0.0.1/ask')
def chat(query: Query):
    response = llm.invoke(query.prompt)
    return {'response': response}

@app.post('/api/v0.0.1/search')
def search(q: str):
    
    search_results = web_searcher.search_web(q)

    return search_results

@app.post('/api/v1/agent/chat/{userid}')
def agent_chat(query: Query, userid: str, session_id: Union[str, None] = None):

    agent = AgentWithSearchTool(userid, session_id)
    response = agent.chat(query.prompt)
    return response

@app.get('/api/v1/agent/messages/{userid}')
def agent_messages(userid: str):
    messages = redis_client.get_messages(userid)

    return messages