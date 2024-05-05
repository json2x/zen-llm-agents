import os
import json
import datetime
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from collections import defaultdict
from typing import List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langgraph.graph import MessageGraph, END

from langchain_mongodb import MongoDBChatMessageHistory
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv()

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    confidence_score: str = Field(description="Assign a confidence score (0-100) indicating your certainty in the answer. Answer with a NUMBER ONLY.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    @traceable
    def respond(self, state: List[BaseMessage]):
        response = []
        for _ in range(3):
            try:
                print("state:", state, flush=True)

                response = self.runnable.invoke({"messages": state})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response
    
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )

class ChekoChatHistory:
    def __init__(self, session_id: str, 
        mongodb_connection_string: Union[str, None] = None, 
        mongodb_database_name: Union[str, None] = None, 
        mongodb_collection_name: Union[str, None] = None
    ):
        
        self.session_id = session_id
        mongodb_connection_string = mongodb_connection_string if mongodb_connection_string else os.getenv("MONGODB_CONNECTION_STRING")
        mongodb_database_name = mongodb_database_name if mongodb_database_name else os.getenv("MONGODB_DATABASE_NAME")
        mongodb_collection_name = mongodb_collection_name if mongodb_collection_name else os.getenv("MONGODB_COLLECTION_NAME")

        assert mongodb_connection_string is not None, "missing MONGODB_CONNECTION_STRING in your environment variables"
        assert mongodb_database_name is not None, "missing MONGODB_DATABASE_NAME in your environment variables"
        assert mongodb_collection_name is not None, "missing MONGODB_COLLECTION_NAME in your environment variables"

        self.history = MongoDBChatMessageHistory(session_id=session_id, 
                                                 connection_string=mongodb_connection_string, 
                                                 database_name=mongodb_database_name, 
                                                 collection_name=mongodb_collection_name)
        
    def messages(self) -> List[BaseMessage]:
        return self.history.messages

    def add_message(self, message: BaseMessage):
        self.history.add_message(message)


class ChekoChatAgent:
    def __init__(self, memory = None, max_iterations: int = 3):
        self.memory = memory
        self.chat_history = []
        self.graph = None
        self.model = ChatOpenAI()
        self.max_iterations = max_iterations

        self.first_instructions = "Provide a detailed ~250 word answer."
        self.revise_instructions = """Revise your previous answer using the new information.
        - You should use the previous critique to add important information to your answer.
            - You MUST provide an APA (American Psychological Association) citation if possible, for example: (Smith, 2019).
            - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In numbered markdown format.
            - You should use the previous critique to remove superfluous information from your answer.
        """

        self.agent_prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a friendly academic assistant named Cheko AI.
                You help students answer questions and provide information from reliable sources.
                Current time: {time}
                
                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Assign a confidence score (0-100) indicating your certainty in the answer. Dont be too generous.
                4. Recommend search queries to research information and improve your answer""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Answer the user's latest question above using the required format. Take into account the entire messages in your response."),
        ]).partial(
            time=lambda: datetime.datetime.now().isoformat()
        )

        self.initial_answer_chain = self.agent_prompt_template.partial(
            first_instruction=self.first_instructions
        ) | self.model.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        self.initial_answer_validator = PydanticToolsParser(tools=[AnswerQuestion])

        self.revision_chain = self.agent_prompt_template.partial(
            first_instruction=self.revise_instructions
        ) | self.model.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
        self.revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

        self.original_generator = ResponderWithRetries(runnable=self.initial_answer_chain, validator=self.initial_answer_validator)
        self.revision_generator = ResponderWithRetries(runnable=self.revision_chain, validator=self.revision_validator)

        self.search_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(), max_results=5)

        # This a helper class we have that is useful for running tools
        # It takes in an agent action and calls that tool and returns the result
        self.tool_executor = ToolExecutor([self.search_tool])
        # Parse the tool messages for the execution / invocation
        self.parser = JsonOutputToolsParser(return_id=True)

        self.graph = self.compile()

    # Define the function that determines whether to continue or not
    def should_continue(self, state: List[BaseMessage]) -> str:
        print("should_continue", state, flush=True)

        last_message: AIMessage = state[-1]
        parsed_tool_calls = self.parser.invoke(last_message)
        # If confidence level is >= 90, then we finish
        for parsed_call in parsed_tool_calls:
            print('parsed_call:', parsed_call, flush=True)
            if int(parsed_call["args"]["confidence_score"]) >= 90:
                return END
            if len(parsed_call["args"]["search_queries"]) == 0:
                return END
        # Otherwise, we execute_tools
        else:
            return "execute_tools"
        
    def execute_tools(self, state: List[BaseMessage]) -> List[BaseMessage]:
        tool_invocation: AIMessage = state[-1]
        print("execute_tools", tool_invocation, flush=True)

        parsed_tool_calls = self.parser.invoke(tool_invocation)
        ids = []
        tool_invocations = []
        for parsed_call in parsed_tool_calls:
            if len(parsed_call["args"]["search_queries"]) == 0:
                return END
            
            for query in parsed_call["args"]["search_queries"]:
                tool_invocations.append(
                    ToolInvocation(
                        # We only have this one for now. Would want to map it
                        # if we change
                        tool="tavily_search_results_json",
                        tool_input=query,
                    )
                )
                ids.append(parsed_call["id"])

        outputs = self.tool_executor.batch(tool_invocations)
        outputs_map = defaultdict(dict)
        for id_, output, invocation in zip(ids, outputs, tool_invocations):
            outputs_map[id_][invocation.tool_input] = output

        return [
            ToolMessage(content=json.dumps(query_outputs), tool_call_id=id_)
            for id_, query_outputs in outputs_map.items()
        ]
    
    # Looping logic:
    def _get_num_iterations(self, state: List[BaseMessage]):
        i = 0
        for m in state[::-1]:
            if not isinstance(m, (ToolMessage, AIMessage)):
                break
            i += 1
        return i

    def event_loop(self, state: List[BaseMessage]) -> str:
        last_message = state[-1]
        print("event_loop", last_message, flush=True)

        parsed_tool_calls = self.parser.invoke(last_message)
        # If confidence level is >= 90, then we finish
        for parsed_call in parsed_tool_calls:
            if int(parsed_call["args"]["confidence_score"]) >= 90:
                return END
            
        # in our case, we'll just stop after N plans
        num_iterations = self._get_num_iterations(state)
        if num_iterations > self.max_iterations:
            return END
        return "execute_tools"
    
    def compile(self):
        builder = MessageGraph()
        builder.add_node("draft", self.original_generator.respond)
        builder.add_node("execute_tools", self.execute_tools)
        builder.add_node("revise", self.revision_generator.respond)

        
        builder.add_edge("execute_tools", "revise")
        builder.add_conditional_edges("draft", self.should_continue)
        builder.add_conditional_edges("revise", self.event_loop)

        builder.set_entry_point("draft")

        if self.memory:
            return builder.compile(checkpointer=self.memory)
        return builder.compile()
    
    def input(self, message: str, session_id: str):
        thread = {"configurable": {"thread_id": session_id}}
        message = [HumanMessage(content=message)]
        for event in self.graph.stream(message, thread):
            for v in event.values():
                print(v)

    def send_message(self, message: str):
        input_message = [HumanMessage(content=message)]
        response = self.graph.invoke(input_message)
        
        return response
    
    def chat(self, message: str, chat_history: ChekoChatHistory):
        input_message = HumanMessage(content=message)
        chat_history.add_message(input_message)

        self.max_iterations += (len(chat_history.messages()))
        messages = self.graph.invoke(chat_history.messages())
        response = "The model seem to have some issues. Please try again later."
        
        # reverse loop through the response and get the last AIMessage
        rev_messages = messages[::-1]
        for message in rev_messages:
            if isinstance(message, AIMessage):
                for tool in message.tool_calls:
                    if tool.get("name") == "AnswerQuestion" \
                    or tool.get("name") == "ReviseAnswer":
                        response = tool.get("args").get("answer")
                        chat_history.add_message(AIMessage(content=response))
                        break
                break

        return {
            "session_id": chat_history.session_id,
            "output": response
        }
        