import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

from langchain.memory import ConversationBufferMemory

from langchain_mongodb import MongoDBChatMessageHistory
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)

import uuid

class AgentWithSearchTool:

    def __init__(self, userid: str, session_id: str = None):
        assert userid is not None, "userid is required"

        if session_id is None:
            session_id = str(uuid.uuid4())

        self.session_id = session_id
        self.cache_session_id = f"{userid}:{self.session_id}"
        
        self.chat_history = []

        self.model = ChatOpenAI(
            model='gpt-3.5-turbo-1106',
            temperature=0.7
        )

        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

        assert url is not None, "UPSTASH_REDIS_REST_URL is required"
        assert token is not None, "UPSTASH_REDIS_REST_TOKEN is required"

        self.history = UpstashRedisChatMessageHistory(url=url, token=token, ttl=3600, session_id=self.cache_session_id)


        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=self.history,
        )


        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly academic assistant named Zenner AI. 
             You help students answer questions and provide reliable academic information. 
             If your answer to the question is sourced from the search tool, observe the following rules:\n\n
             [RULES]\n
             1. Always Provide an APA (American Psychological Association) citation if possible, for example: (Smith, 2019).\n
             2. Provide links where you get the information at the end of your response.\n\n
             If you dont know the answer to the question, respond politely and ask them to try rephrasing the question.\n\n
             Suggest the follwing when rephrasing the question:\n\n
             [SUGGESTED TERMS]\n
             1. Use terms like search, find, look up, etc.\n
             2. Add year, dates and time related terms like now, latest, etc to fetch the most relevant information.
             
             And lastly, if you're ask how to donate to this project, ask them first where is most convenient to donate, gcash or buymeacoffee.
             Send the following markdown message base on their answer.
             GCash: ![gcash](https://www.studyassitant.com/assets/gcash_qr.3c44bdd9.png) 
             Buymecoffee: ![buymeacoffee](https://www.studyassitant.com/assets/bmc_qr.9c9b87f3.png)
             """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])


        self.search_tool = TavilySearchResults()
        self.tools = [self.search_tool]

        self.agent = create_openai_functions_agent(
            llm=self.model,
            prompt=self.prompt,
            tools=self.tools
        )

        self.agentExecutor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory
        )

    def chat(self, user_input):
        response = self.agentExecutor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response["output"]))
        return {"session_id": self.session_id, "response": response["output"]}