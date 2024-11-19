import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Define las rutas del modelo y el archivo PDF
llama_model_name = "meta-llama/Llama-3.2-1B"  # Verifica que tienes acceso
pdf_path = "/Users/mau/Desktop/proyecto/herramientas/Back_Herramientas/src/recursos.pdf"

# Cargar el modelo Llama y tokenizer
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
model = AutoModelForCausalLM.from_pretrained(llama_model_name)
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Cargar el modelo de Sentence Transformer para la indexación de oraciones
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Indexar el contenido del PDF usando FAISS
def index_pdf_content(text):
    sentences = text.split(". ")
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    embeddings = embeddings.cpu().detach().numpy()

    # Crear índice FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return sentences, index

# Función para hacer preguntas al chatbot
def ask_question(pdf_path, question):
    # Extraer e indexar texto del PDF
    text = extract_text_from_pdf(pdf_path)
    sentences, index = index_pdf_content(text)

    # Generar embedding para la pregunta
    question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().detach().numpy()

    # Buscar las oraciones más similares
    D, I = index.search(question_embedding, k=3)
    context = " ".join([sentences[i] for i in I[0]])

    # Generar respuesta usando el modelo Llama
    prompt = f"Contexto: {context}\nPregunta: {question}\nRespuesta:"
    response = qa_pipeline(prompt, max_length=150, num_return_sequences=1)
    
    return response[0]["generated_text"]

# Ejemplo de uso
question = "¿que es una red neuronal?"
answer = ask_question(pdf_path, question)
print(answer)
