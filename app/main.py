import ollama
from flask import Flask, jsonify, request

app = Flask(__name__)


modelo = "llama3.2"

historia = [{'role':"system","content":"Eres una IA, para responder sola y unicamente preguntas de la clase de herramientas para el procesamiento de IA"}]



@app.post("/mensaje")
def init():
    data = request.get_json()
    mensaje = data["mensaje"]

    historia.append({'role':"user","content":mensaje})
    respuestas = ollama.chat(model = modelo, messages=historia)
    historia.append({'role':"system","content":respuestas["message"]["content"]})

   

    return jsonify({"respuesta": respuestas["message"]["content"]}), 200


if __name__ == "__main__":
    app.run(debug=True)









