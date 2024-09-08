import os
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Set up LLM and API Wrappers
LLM = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Wikipedia and Arxiv Setup
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=100)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

# Step 2: Web Document Loader and FAISS Setup
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)

vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(
    retriever, "langsmith_search", "Search for information about LangSmith"
)

# Step 3: Set up Agent and Tools
tools = [wiki_tool, retriever_tool, arxiv_tool]
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(LLM, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 4: Invoke the Agent with a Query
response = agent_executor.invoke({"input": "Tell me about LangSmith"})
print(response["output"])
