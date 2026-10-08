from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("AI Chatbot")
st.markdown("This is a simple AI chatbot using Google Gemini model.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask me anything:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    response = llm.invoke(query)
    st.chat_message("AI").write(response.text())
    st.session_state.messages.append({"role": "AI", "content": response.text()})

