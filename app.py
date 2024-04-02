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

def main(): 
    gui.setup(wide=True, text="Embed Document Chunks")
    st.title('Embed Documents')
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

    if st.button("Clear the embeddings"):
        sql.query("DELETE FROM embeddings;")
        sql.connection.commit()
        st.write("Document embeddings cleared.")

    if st.button("Example Embedding"):
        with st.spinner("Embedding..."):
            start_time = time.time()  # start timing
            embedding = ollama.embeddings(model='mistral', prompt='The sky is blue and there are 20 trees ')
            end_time = time.time()  # end timing
            elapsed_time = end_time - start_time  # calculate elapsed time
            st.write(f"Time taken: {elapsed_time} seconds")  # print elapsed time
            st.write(embedding)

    if not st.button("Embed each chunk?"):
        st.stop()

    chunks = sql.query("SELECT * FROM chunks;")
    chunks_df = pd.DataFrame(chunks)
    st.dataframe(chunks_df)
    for index, chunk in chunks_df.iterrows():
        content = chunk['content']
        with st.spinner("Embedding..."):
            start_time = time.time()
            embedding = st.session_state.client.embeddings(model='mistral', prompt=content)
            embedding_str = ', '.join(map(str, embedding['embedding']))  # convert numpy array or list to string
            end_time = time.time()
            elapsed_time = end_time - start_time
            
        # the SQL database needs source, chunk_id, and embedding
        st.write(f"Chunk {chunk['chunk_number']} embedded from {chunk['source']} in {elapsed_time} seconds")
        sql.cursor.execute(f"INSERT INTO embeddings (source, chunk_number, embedding) VALUES ('{chunk['source']}', '{chunk['chunk_number']}', '{embedding_str}');")
        sql.connection.commit()
        st.write("Embedded.")

main()