import argparse
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from dotenv import load_dotenv
import streamlit as st

# app config
st.set_page_config(page_title="Bible Bot", page_icon="🤖")
st.title("Bible Bot")

load_dotenv()
CHROMA_PATH = "chroma"
OpenAI_key = os.environ.get("OPEN_AI_KEY")



def main(user_query,chat_history):
    # Create CLI.
    #parser = argparse.ArgumentParser()
    #parser.add_argument("query_text", type=str, help="The query text.")
    #args = parser.parse_args()
    #query_text = args.query_text
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    Chat History: {chat_history}

    Context: {context}

    ---

    Answer the question based on the above context: {question}
    """

    embedding_function = OpenAIEmbeddings()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #results = db.similarity_search_with_relevance_scores(query_text, k=10)
    results= db.as_retriever(search_kwargs={"k":10})

    
    bm25_retriever = BM25Retriever.from_texts(db.get(["metadatas", "documents"]))
    bm25_retriever.k = 10


    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, results],
                                       weights=[0.5, 0.5])

    
    context_text= ensemble_retriever.get_relevant_documents(user_query)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(chat_history= chat_history,context=context_text, question=user_query)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    print("Response from model:", response_text)            

    #sources = [doc.metadata.get("source", None) for doc, _score in results]
    #formatted_response = f"Response: {response_text}" #\nSources: {sources}
    return  response_text
    
# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":

    # Append Human message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Get AI response
    response = main(user_query, st.session_state.chat_history)

    # Check if the response is valid (non-empty string)
    if response and isinstance(response, str):
        # Append AI response to chat history if it's valid
        st.session_state.chat_history.append(AIMessage(content=response))

        # Display the response
        with st.chat_message("AI"):
            st.markdown(response)
    else:
        # Handle the case when the response is invalid
        st.session_state.chat_history.append(AIMessage(content="Sorry, I couldn't generate a valid response."))
        with st.chat_message("AI"):
            st.markdown("Sorry, I couldn't generate a valid response.")