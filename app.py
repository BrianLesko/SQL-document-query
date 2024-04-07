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
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def setup_session_state():
    # Keep the connection to the LLM accross reruns
    if "client" not in st.session_state:
        st.session_state.my_ollama = ollamaInterface.ollamaInterface()
        message = st.session_state.my_ollama.start_container()
        st.write(message)
        st.session_state.client = Client(host='http://localhost:11434')
        st.session_state.my_ollama.start()
        st.session_state.messages = []
    # Keep the connection to the SQL container accross reruns
    try:
        if "sql" not in st.session_state:
            st.session_state.sql = SQLConnect()
            st.session_state.sql.connect()
        if "sql" in st.session_state:
            sql = st.session_state.sql
    except Exception as e:
        st.error(e)
        st.error("Could not connect to the SQL container. Docker may not be running.")
        st.stop()

def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            col1, col2, col3 = st.columns([1,69,1])
            with col2:
                if message['role'] in ['user', 'assistant']:
                    st.caption("You" if message['role'] == 'user' else "Mistral 7B")
                    st.write(message['content'])
                    if 'num_tokens' in message and message['role'] == 'user':
                        st.caption(f"Number of tokens: {message['num_tokens']}")
                    if 'response_time' in message and message['role'] == 'assistant':
                        st.caption(f"Response time: {message['response_time']:.1f}s")

def process_stream(stream, text, begin_time, Time, placeholder):
    times = []
    for chunk in stream:
        with Time: 
            times.append(time.time()-begin_time)
            st.caption(f"Time: {times[-1]:.1f}s")
        text += chunk['message']['content']
        wrapped_text = textwrap.fill(text, width=80)
        placeholder.markdown(f'<p style="font-size:16px">{wrapped_text}</p>', unsafe_allow_html=True)
        if 'assistant' in st.session_state.messages[-1]['role']:
            st.session_state.messages[-1] = {
                'role': 'assistant', 
                'content': wrapped_text, 
                'response_time': times[-1]
            }
        else:
            st.session_state.messages.append({
                'role': 'assistant', 
                'content': wrapped_text, 
                'response_time': times[-1]
            })

def get_similarity(embedding1, embedding2):
    embedding1_2d = np.array(embedding1).reshape(1, -1)
    embedding2_2d = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(embedding1_2d, embedding2_2d)
    return similarity[0][0]

def main(): 
    gui.setup(wide=True, text="Query Documents")
    st.title('Query documents')
    current_task = st.empty()

    setup_session_state()
    sql = st.session_state.sql

    with st.sidebar:
        result = st.session_state.sql.get_tables()
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

    display_messages()

    with st.sidebar:
        n = st.number_input("Number of sources to display", min_value=1, max_value=12, value=4)

    Time = st.empty()
    query = st.chat_input("Enter a query:")
    if query: 
        num_tokens = len(tokenizer(query)['input_ids'])
        start_time = time.time()
        with st.chat_message("user"): 
            col1, col2, col3 = st.columns([1,69,1])
            with col2:
                st.session_state.messages.append({'role': 'user', 'content': query, 'num_tokens': len(tokenizer(query)['input_ids']), 'response_time': time.time() - start_time})
                st.caption(" You")
                st.write(query)
                st.caption(f"Number of tokens: {num_tokens}")

        with st.chat_message("assistant"):
            col1, col2, col3 = st.columns([1,69,1])
            with col2:
                st.caption(f"Mistral 7B")
                with st.status("Searching your dataset..."):  
                    st.write("Embedding your query")      
                    my_embedding = st.session_state.client.embeddings(model='mistral', prompt=query)['embedding']
                    st.write("Retreiving your database embeddings")
                    embeddings = pd.DataFrame(sql.query("SELECT * FROM embeddings;"))
                
                    similarities = []
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
                    st.write(f"Finding the best {n} sources")
                    for similarity, source, chunk_number in top_n_sorted:
                        st.caption(f"Document {source} chunk {chunk_number} has a similarity of {similarity}")
                        text = sql.query(f"SELECT content FROM chunks WHERE source = '{source}' AND chunk_number = {chunk_number};")[0]['content']
                        st.caption(text)
                        documents.append(f"Document {source} chunk {chunk_number} has a similarity of {similarity}")
                        documents.append(text)
                    st.write("Calculed cosine similarity for each document.")
                    documents = ' '.join(documents)

                    input = f"Your goal is to answer this query: ({query}). Use the following sources: {documents}"

                with st.spinner("Querying the model"):
                    begin_time = time.time()
                    messages = [
                        {'role': 'system', 'content': documents},
                        {'role': 'user', 'content': query}
                    ]
                    stream = ollama.chat(model='mistral',messages=messages,stream=True,)
                    with col2:
                        placeholder = st.empty()
                        Time = st.empty()
            
                    text = ""
                    chunk = next(stream)
                    text += chunk['message']['content']
                process_stream(stream, text, begin_time, Time, placeholder)
main()