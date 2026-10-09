import csv
import math
from pathlib import Path
from pydantic import ValidationError
import joblib
from clasificador_paquetes.contracts import Entrada, Salida
 

def leer_csv(ruta: Path) -> list[Entrada]:
    list = []
    with open(ruta, newline="", encoding= "utf-8") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader, start=2):
            try:
                lista = Entrada(**row)
                list.append(lista)
            except ValidationError as error:
                raise ValueError(f"Error en la línea {i}: {error}") from error
    return list


def preprocesar(entrada: Entrada) -> list[float]:
    '''
    Completa preprocesar(entrada). Recibe una Entrada ya validada y devuelve una lista de dos números: primero peso_kg redondeado a una cifra decimal (tip: round) y después distancia_km, sin modificar la distancia. No incluyas el identificador en el vector. No cambies el objeto recibido.

Ejemplo: peso 2.36 y distancia 8 producen [2.4, 8.0].
    '''
    a = entrada.peso_kg 
    agrande = a*10
    aredondeado = math.ceil(agrande)
    areal = aredondeado/10
    entrada.peso_kg = areal

    return [
        entrada.peso_kg,
        entrada.distancia_km
    ]



def cargar_modelo(ruta: Path):
    modelo = joblib.load(ruta)
    return modelo
    

def predecir(entrada: Entrada, modelo) -> Salida:    
    prob_categoria = modelo.predict(entrada)
    probability = modelo.predict_proba(entrada)
    mayor_probability = probability[0][1]

    return Salida(
        identificador= entrada.id_paquete,
        prob_categoria=prob_categoria,
        probabilidad=mayor_probability
    )

    '''
    B5. Modelo e inferencia (1,25 puntos)
Completa cargar_modelo(ruta) y predecir(entrada, modelo).

Carga con joblib el modelo proporcionado
El modelo recibe una matriz (2D): una lista que contiene el vector de dos características, en el orden de B4.
modelo.predict(matriz) devuelve una categoría por fila. modelo.predict_proba(matriz) devuelve las probabilidades en el mismo orden que modelo.classes_.
Para una única entrada, devuelve un objeto Salida con su identificador, la categoría predicha y, como confianza, la mayor probabilidad de esa fila. La salida debe validarse con Pydantic; no devuelvas un diccionario sin validar.'''



def guardar_csv(resultados: list[Salida], ruta: Path) -> None:
    OUTPUT_COLUMNS = ["id_paquete", "CATEGORIA", "Confianza"]
    '''
    Completa guardar_csv(resultados, ruta) y ejecutar(entrada, modelo, salida).

El CSV de salida tendrá cabecera id_paquete,categoria,confianza y una fila por paquete, respetando el orden original.
ejecutar debe leer y validar todas las entradas, cargar el modelo una sola vez, predecir cada paquete y guardar al final.
Si falla una entrada o una predicción, no debe crearse un archivo de resultados parcial. Si la ruta de salida ya existía, su contenido debe permanecer intacto ante ese fallo. No se pide proteger frente a un fallo del disco mientras se escribe.
La ejecución desde la raíz será uv run python -m clasificador_paquetes.inference. El bloque de arranque y las rutas ya están en el starter; no se pide crear argumentos de terminal.
    '''
    with open(ruta, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for i in resultados:
            writer.writerow(i.model_dump())
    return 


def ejecutar(entrada: Path, modelo: Path, salida: Path) -> None:
    lista= preprocesar(entrada)
    predictions = []
    for i in lista:
        prediction = predecir(modelo, i )
        predictions.append(prediction)
    return


if __name__ == "__main__":
    ejecutar(
        Path("data/raw/paquetes.csv"),
        Path("models/modelo.joblib"),
        Path("resultados.csv"),
    )
