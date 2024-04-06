# Brian Lesko
# 3/18/2024

import streamlit as st
from SQLConnect import SQLConnectDocker as SQLConnect
from customize_gui import gui as gui 
gui = gui()
import time
import pandas as pd
from ollama import Client
import ollama
import ollamaInterface
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textwrap

def get_similarity(embedding1, embedding2):
    # Reshape the embeddings to 2D arrays
    embedding1_2d = np.array(embedding1).reshape(1, -1)
    embedding2_2d = np.array(embedding2).reshape(1, -1)

    # Compute the cosine similarity
    similarity = cosine_similarity(embedding1_2d, embedding2_2d)

    # Return the similarity
    return similarity[0][0]

def main(): 
    gui.setup(wide=True, text="Query Documents")
    st.title('Query documents')
    current_task = st.empty()

    # Keep the connection to the SQL container accross reruns
    if "sql" not in st.session_state:
        st.session_state.sql = SQLConnect()
        st.session_state.sql.connect()
    if "sql" in st.session_state:
        sql = st.session_state.sql

    # Keep the connection to the LLM accross reruns
    if "client" not in st.session_state:
        st.session_state.my_ollama = ollamaInterface.ollamaInterface()
        message = st.session_state.my_ollama.start_container()
        st.write(message)
        st.session_state.client = Client(host='http://localhost:11434')
        st.session_state.my_ollama.start()

    with st.sidebar:
        result = sql.get_tables()
        Tables = st.empty()
        with Tables: st.table(result)
        "---"

        # Prep
        sql.query("USE user")
        names = sql.query("SELECT name FROM summary;")
        names_df = pd.DataFrame(names)
        unique_names = names_df['name'].unique()
        st.write(f"{len(unique_names)} Unique entries in the library.")
        content_count = sql.query("SELECT COUNT(*) as count FROM content;")
        st.write(f"{content_count[0]['count']} rows in the content table.")
        if content_count[0]['count'] == unique_names.size:
            st.write("The content table is up to date.")
        else:
            st.write("The content table is not up to date. Please run the 'Add Content' script.")
        num_chunks = sql.query("SELECT COUNT(*) FROM chunks;")
        st.write(f"{num_chunks[0]['COUNT(*)']} Document chunks in the database.")

        # Is the embeddings table up to date?
        num_embeddings = sql.query("SELECT COUNT(*) FROM embeddings;")
        st.write(f"{num_embeddings[0]['COUNT(*)']} Embeddings in the database.")
        if num_embeddings[0]['COUNT(*)'] == num_chunks[0]['COUNT(*)']:
            st.write("The embeddings table is up to date.")
        else:
            st.write("The embeddings table is not up to date. Please run the 'Embed Documents' script.")

    # Display the existing messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for role, message in st.session_state.messages:
        with st.chat_message(role):
            col1, col2, col3 = st.columns([1,69,.1])
            with col2:
                if role == 'user':
                    st.caption("You")
                else:
                    st.caption("Mistral 7B")
                st.write(message)

    query = st.chat_input("Enter a query:")
    if query: 
        with st.chat_message("user"): st.write(query)
        my_embedding = st.session_state.client.embeddings(model='mistral', prompt=query)['embedding']
        embeddings = pd.DataFrame(sql.query("SELECT * FROM embeddings;"))
        
        similarities = []
        n = 2
        top_n = []
        for index, row in embeddings.iterrows():
            embedding = row['embedding'].split(", ")
            source = row['source']
            chunk_number = row['chunk_number']

            similarity = get_similarity(my_embedding, embedding)
            top_n.append((similarity, source, chunk_number))
            if len(top_n) > n:
                top_n.remove(min(top_n, key=lambda x: x[0]))

        top_n_sorted = sorted(top_n, key=lambda x: x[0], reverse=True)
        st.write(f"Searched through {index} documents.")
        documents = []
        with st.expander(f"{n} Sources"):
            for similarity, source, chunk_number in top_n_sorted:
                st.caption(f"Document {source} chunk {chunk_number} has a similarity of {similarity}")
                text = sql.query(f"SELECT content FROM chunks WHERE source = '{source}' AND chunk_number = {chunk_number};")[0]['content']
                st.write(text)
                documents.append(f"Document {source} chunk {chunk_number} has a similarity of {similarity}")
                documents.append(text)
        
        
        documents = ' '.join(documents)
        input = f"Your goal is to answer this query: ({query}). Use the following sources: {documents}"
        with st.spinner("Querying the model"):
            stream = ollama.chat(model='mistral',messages=[{'role': 'user', 'content': input}],stream=True,)
        with st.chat_message("assistant"):
            col1, col2, col3 = st.columns([1,69,1])
            with col2:
                st.caption("Mistral 7B")
                placeholder = st.empty()
            text = ""
            for chunk in stream:
                text += chunk['message']['content']
                wrapped_text = textwrap.fill(text, width=80)
                placeholder.markdown(f'<p style="font-size:16px">{wrapped_text}</p>', unsafe_allow_html=True)
        

main()