import os
import time
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from pinecone import ServerlessSpec, Pinecone
from langchain import hub



class Agent:
    chat_history: list = []
    max_chat_history: int = 10
    def __init__(self):
        # initialize embedding model
        self.embed = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index = self._init_index()
        self.agent_executor = self._init_agent()

    def __call__(self, text: str):
        output = self.agent_executor.invoke({
            "input": text,
            "chat_history": self.chat_history
        })
        # add user input and response to chat history
        self._add_response(
            human_message=text,
            ai_message=output["output"]
        )
        return output["output"]

    def _init_index(self):
        # initialize pinecone index
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        spec = ServerlessSpec(cloud="aws", region="us-west-2")
        self.index_name = "rag-chatbot"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                self.index_name,
                dimension=self._get_embedding_size(),
                metric="dotproduct",
                spec=spec
            )
            # wait for index to be initialized
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        # connect to index
        return self.pc.Index(self.index_name)

    def _get_embedding_size(self):
        vec = self.embed.embed_query("ello")
        return len(vec)
    
    def _init_agent(self):
        prompt = hub.pull("hwchase17/openai-functions-agent")
        # initialize chat LLM
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0.0
        )
        tools = [self.tool_arxiv_search]
        agent = create_openai_tools_agent(
            prompt=prompt,
            llm=llm,
            tools=tools
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )
        return agent_executor

    def _add_response(self, human_message: str, ai_message: str):
        # add new message
        self.chat_history.extend([
            HumanMessage(content=human_message),
            AIMessage(content=ai_message)
        ])
        # limit chat message to most recent self.max_chat_history messages
        self.chat_history = self.chat_history[-self.max_chat_history:]

    @tool
    def tool_arxiv_search(self, query: str) -> str:
        """Use this tool when answering questions about AI, machine learning, data
        science, or other technical questions that may be answered using arXiv
        papers.
        """
        # create query vector
        xq = self.embed.embed_query(query)
        # perform search
        out = self.index.query(vector=xq, top_k=5, include_metadata=True)
        # reformat results into string
        results_str = "\n\n".join(
            [x["metadata"]["text"] for x in out["matches"]]
        )
        return results_str