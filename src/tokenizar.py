from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import PyPDF2
import numpy as np
import os
import textwrap

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

pdf_dir = "/Users/mau/Desktop/proyecto/herramientas/Back_Herramientas/src"
pdf_texts = []
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_dir, filename)
        pdf_text = extract_text_from_pdf(file_path)
        pdf_texts.append(pdf_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(pdf_texts).toarray()
dimension = X.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(X.astype('float32'))

def chatbot(question):
    question_vector = vectorizer.transform([question]).toarray().astype('float32')
    _, I = index.search(question_vector, k=40) 
    fragment_length = 4500  

    best_answer = None
    best_score = 0
    context_text = ""
    for i in I[0]:
        matched_text = pdf_texts[i]
        context_text += " " + matched_text[:fragment_length]
    fragments = textwrap.wrap(context_text, fragment_length)

    for fragment in fragments:
        result = qa_pipeline(question=question, context=fragment)
        if len(result['answer']) > 15 and result['score'] > best_score:
            best_answer = result['answer']
            best_score = result['score']
    if best_answer:
        response = best_answer
    else:
        response = "No se encontró una respuesta relevante en el documento."
    
    return f"Respuesta: {response}"

if __name__ == "__main__":

    question = input("Haz tu pregunta: ")
    print(chatbot(question))
