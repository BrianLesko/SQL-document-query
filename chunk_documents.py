# Brian lesko 

# Chunk documents into smaller pieces, but keep sentences intact. 

from transformers import AutoTokenizer
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def chunk_document(tokens, min_chunk_size, max_chunk_size):
    document_chunks = []
    start_index = 0

    while start_index < len(tokens):
        overlap = 0
        # Get a chunk starting from the current start_index
        chunk, end_index, period_ind = create_chunk(tokens[start_index:], min_chunk_size, max_chunk_size)
        if len(period_ind) > 1: # change overlap here
            #st.write(f"Period indices: {period_ind}")
            overlap = period_ind[-1] - period_ind[-2]


        # Add the chunk to the list
        document_chunks.append(chunk)

        # Update the start index for the next chunk
        start_index += end_index - overlap

        # Check if we have reached the end of the tokens
        if end_index == 0:
            break

    return document_chunks

# Helper function to create a chunk
def create_chunk(tokens, min_chunk_size, max_chunk_size):
    chunk = []
    end_index = 0
    period_indices = []

    for index, token in enumerate(tokens):
        chunk.append(token)
        if token == '.':
            period_indices.append(index)
        if len(chunk) >= min_chunk_size:
            if token == '.' and len(chunk) >= min_chunk_size:
                end_index = index + 1
                break
            elif len(chunk) >= max_chunk_size:
                end_index = index + 1
                break

    chunk_ids = tokenizer.convert_tokens_to_ids(chunk)
    chunk_text = tokenizer.decode(chunk_ids)

    return chunk_text, end_index, period_indices


