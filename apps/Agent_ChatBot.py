from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", streaming=True)
search = GoogleSerperAPIWrapper()


if "memory" not in st.session_state:
    st.session_state.memory= MemorySaver()
    st.session_state.history = []


agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer = st.session_state.memory,
    system_prompt="You are a agent and can search for any question on google."
)

st.subheader("QuickAnswer - Answers at the speed of thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query=st.chat_input("ASk Anything? ")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({
        "role": "user",
        "content": query
    })

    response = agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="messages"
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk, metadata in response:
            if isinstance(chunk, str):
                message += chunk
            elif hasattr(chunk, "content"):
                content = chunk.content

                if isinstance(content, str):
                    message += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            message += item.get("text", "")

            space.markdown(message)

        st.session_state.history.append({"role": "ai", "content": message})