import streamlit as st

# st.title("Netflix Title Generator")
# st.write("This is the Netflix Title Generator model page.")

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to capitalize title
def capitalize_title(title):
    return ' '.join(word.capitalize() for word in title.split())

# Load model and tokenizer (caching this to avoid reloading every time)
@st.cache_resource
def load_model():
    model_dir = r"C:\Users\ronit\OneDrive\Desktop\College\Sem_7\NNDL\NNDL_Repo\DL_MiniProject\Title_Model"
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)  # Move model to device
    return model, tokenizer

# Function to generate titles
def generate_title(description, model, tokenizer, max_length=20, num_return_sequences=5):
    model.eval()
    input_text = "summarize: " + description
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=num_return_sequences,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    generated_titles = [capitalize_title(tokenizer.decode(output, skip_special_tokens=True)) for output in outputs]
    return generated_titles

# Function to select best title based on cosine similarity
def select_best_title(description, titles):
    texts = [description] + titles
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_index = cosine_similarities.argmax()
    return titles[best_index]

# Streamlit application interface
st.title("Netflix Title Generator")

# Load the model and tokenizer
model, tokenizer = load_model()

# Get user input description
description = st.text_area("Enter the movie/show description:", height=300)

if st.button("Generate Title"):
    if description:
        # Generate and display titles
        generated_titles = generate_title(description, model, tokenizer)
        best_title = select_best_title(description, generated_titles)
        
        # st.write("Generated Titles:")
        # for i, title in enumerate(generated_titles, 1):
        #     st.write(f"{i}. {title}")
        
        st.write(f"\nBest Generated Title: {best_title}")
    else:
        st.write("Please enter a description.")
