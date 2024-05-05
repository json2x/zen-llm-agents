from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


class RetrieverChain:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self.history = []
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        self.embeddings = OpenAIEmbeddings()
        self.webpages = self.get_webpages()
        self.split_docs = self.split_documents()
        self.vector_store = self.set_vector_store()
        self.retrieval_chain = self.create_chain()

    def get_webpages(self):
        loader = WebBaseLoader(self.urls)
        webpages = loader.load()
        return webpages

    def split_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        split_docs = splitter.split_documents(self.webpages)
        return split_docs

    def set_vector_store(self):
        vector_store = FAISS.from_documents(documents=self.split_docs, embedding=self.embeddings)
        return vector_store
    
    def append_chat_history(self, human_message: str = '', ai_message: str = ''):
        if human_message:
            self.history.append(HumanMessage(content=human_message))
        
        if ai_message:
            self.history.append(AIMessage(content=ai_message))

    def create_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise.
                
                Context: {context}
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )

        # Replace retriever with history aware retriever
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # retriever_prompt = ChatPromptTemplate.from_messages([
        #     MessagesPlaceholder(variable_name="chat_history"),
        #     ("user", "{input}"),
        #     ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        # ])
        retriever_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, 
            rephrase the follow up question to be a standalone question.
            \n\nChat History:\n{chat_history}
            \nFollow Up Input: {input}
            \nStandalone Question:"""
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=retriever_prompt
        )

        retrieval_chain = create_retrieval_chain(
            # retriever, Replace with History Aware Retriever
            history_aware_retriever,
            chain
        )

        return retrieval_chain