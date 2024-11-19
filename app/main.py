from transformers import pipeline
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

context = r"""
La Universidad Iberoamericana León es una universidad privada confiada a la Compañía de Jesús, ubicada en la ciudad de León, Guanajuato. Pertenece al Sistema Universitario Jesuita y a la AUSJAL. Fue fundada en 1978 para extender los servicios educativos de la Ibero a la región del Bajío, siendo así el primer campus al interior de la República.
"""

result = question_answerer(question="Donde se ubica la universidad?",     context=context)
print(
f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
)