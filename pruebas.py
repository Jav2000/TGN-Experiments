import pickle

# Abre el archivo .pkl en modo lectura binaria
with open('./data/tgbl-wiki-v2/tgbl-wiki_test_ns_v2.pkl', 'rb') as f:
    # Carga el contenido del archivo en un objeto Python
    data = pickle.load(f)

# Verifica si el diccionario no está vacío
if data:
    # Obtiene la primera entrada del diccionario
    primera_clave = next(iter(data))
    primer_valor = data[primera_clave]

    # Imprime la primera entrada del diccionario
    print("Primera entrada del diccionario:")
    print(f"Clave: {primera_clave}")
    print(f"Valor: {primer_valor}")
else:
    print("El diccionario está vacío.")