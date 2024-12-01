import ollama
from flask import Flask, jsonify, request
import pdfplumber


app = Flask(__name__)







# def extraer_texto_pdf(ruta_pdf):
#     texto = ""
#     with pdfplumber.open(ruta_pdf) as pdf:
#         for pagina in pdf.pages:
#             texto += pagina.extract_text()

#     print("Texto extraído del PDF:")
#     print(texto)
#     return texto



#texto_pdf = extraer_texto_pdf("src/recursos.pdf")




modelo = "llama3.2"

# historia = [
#     {'role': 'system', 'content': f"Eres una IA, responde solo preguntas basándote en el siguiente texto: {texto_pdf}, sé breve y solo responde, segun el texto"}
# ]

historia = [
    {'role': 'system', 'content': f"Eres una IA, responde que eres experto en la Materia de Herramientas Para Ingenieria en I.A, sé breve y solo responde preguntas de esa indole, no más"}
]




@app.post("/mensaje")
def init():
    data = request.get_json()
    mensaje = data["mensaje"]

    print(mensaje)

    historia.append({'role':"user","content":mensaje})
    respuestas = ollama.chat(model = modelo, messages=historia)
    historia.append({'role':"system","content":respuestas["message"]["content"]})

   

    return jsonify({"respuesta": respuestas["message"]["content"]}), 200


if __name__ == "__main__":
    app.run(debug=True)









