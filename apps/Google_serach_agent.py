from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()


agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer = memory,
    system_prompt="You are a agent and can search for any question on google."
)


while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("Good Bye")
        break


    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        )
    
    print("AI: ", response["messages"][-1].content[0]["text"])

