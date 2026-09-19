from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
#tavily search is better than custom tool
#from langchain_tavily import TavilySearch

from tavily import TavilyClient

tavily = TavilyClient()

### User defined tool ###
@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatDeepSeek(temperature=0, model="deepseek-v4-flash")
#llm = ChatOllama(temperature=0, model="gpt-oss:20b")
tools = [search]
# tools=[TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings in linked in for an AI engineer using langchain in Dhaka or remote allowing people living in dhaka")})
    print(result)


if __name__ == "__main__":
    main()
