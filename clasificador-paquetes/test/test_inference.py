"""Pruebas de los requisitos B2-B6. Ejecutar desde la raíz: uv run pytest."""
import csv
from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from clasificador_paquetes.contracts import Entrada, Salida
from clasificador_paquetes import inference

ROOT = Path(__file__).resolve().parents[1]


def valido(**cambios):
    datos = {"id_paquete": " P01 ", "peso_kg": "2.36", "distancia_km": "8"}
    datos.update(cambios)
    return Entrada.model_validate(datos)


def escribir_entrada(ruta, filas):
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["id_paquete", "peso_kg", "distancia_km"])
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


class ModeloPrueba:
    """Doble de prueba con la misma interfaz que el modelo suministrado."""
    classes_ = np.array(["normal", "urgente"])

    def __init__(self, categoria="urgente", probabilidades=(0.23, 0.77)):
        self.categoria = categoria
        self.probabilidades = probabilidades
        self.matrices_predict = []
        self.matrices_proba = []

    def predict(self, matriz):
        self.matrices_predict.append(matriz)
        return np.array([self.categoria])

    def predict_proba(self, matriz):
        self.matrices_proba.append(matriz)
        return np.array([self.probabilidades])


# B2: contratos, conversión de texto numérico, límites y campos extra.
def test_entrada_valida():
    entrada = valido()
    assert entrada.id_paquete == "P01"
    assert entrada.peso_kg == pytest.approx(2.36)
    assert entrada.distancia_km == pytest.approx(8)
    assert isinstance(entrada.peso_kg, float)
    assert isinstance(entrada.distancia_km, float)


@pytest.mark.parametrize("cambios", [
    {"id_paquete": ""}, {"id_paquete": " \t\n "}, {"id_paquete": None},
    {"peso_kg": 0}, {"peso_kg": -1}, {"peso_kg": 30.01},
    {"peso_kg": "abc"}, {"peso_kg": "nan"}, {"peso_kg": "inf"},
    {"peso_kg": "-inf"}, {"peso_kg": None},
    {"distancia_km": -0.01}, {"distancia_km": 200.01},
    {"distancia_km": "abc"}, {"distancia_km": "nan"},
    {"distancia_km": "inf"}, {"distancia_km": "-inf"},
    {"distancia_km": None}, {"extra": 1},
])
def test_entrada_invalida(cambios):
    with pytest.raises(ValidationError):
        valido(**cambios)


@pytest.mark.parametrize("campo", ["id_paquete", "peso_kg", "distancia_km"])
def test_entrada_campo_obligatorio(campo):
    datos = {"id_paquete": "P01", "peso_kg": "1", "distancia_km": "2"}
    del datos[campo]
    with pytest.raises(ValidationError):
        Entrada.model_validate(datos)


@pytest.mark.parametrize("peso,distancia", [(30, 200), (0.1, 0), ("30", "200"), ("0.1", "0")])
def test_limites_entrada(peso, distancia):
    entrada = valido(peso_kg=peso, distancia_km=distancia)
    assert entrada.peso_kg == float(peso)
    assert entrada.distancia_km == float(distancia)


@pytest.mark.parametrize("categoria,confianza", [("normal", 0), ("normal", 1), ("urgente", 0), ("urgente", 1)])
def test_salida_valida(categoria, confianza):
    salida = Salida(id_paquete="P01", categoria=categoria, confianza=confianza)
    assert salida.categoria == categoria
    assert salida.confianza == confianza


@pytest.mark.parametrize("cambios", [
    {"id_paquete": ""}, {"id_paquete": None}, {"categoria": "otra"},
    {"categoria": None}, {"confianza": -0.01}, {"confianza": 1.01},
    {"confianza": "nan"}, {"confianza": "inf"}, {"confianza": "-inf"},
    {"confianza": "abc"}, {"confianza": None}, {"extra": 1},
])
def test_salida_invalida(cambios):
    datos = {"id_paquete": "P01", "categoria": "normal", "confianza": 0.5}
    datos.update(cambios)
    with pytest.raises(ValidationError):
        Salida.model_validate(datos)


@pytest.mark.parametrize("campo", ["id_paquete", "categoria", "confianza"])
def test_salida_campo_obligatorio(campo):
    datos = {"id_paquete": "P01", "categoria": "normal", "confianza": 0.5}
    del datos[campo]
    with pytest.raises(ValidationError):
        Salida.model_validate(datos)


# B3: lectura validada, orden, filas incorrectas y CSV vacío.
def test_carga_csv_suministrado():
    entradas = inference.leer_csv(ROOT / "data/raw/paquetes.csv")
    assert len(entradas) == 6
    assert all(isinstance(entrada, Entrada) for entrada in entradas)
    assert [entrada.id_paquete for entrada in entradas] == ["P01", "P02", "P03", "P04", "P05", "P06"]


def test_carga_csv_orden_y_valores(tmp_path):
    ruta = escribir_entrada(tmp_path / "entrada.csv", [
        {"id_paquete": " Z09 ", "peso_kg": "3.27", "distancia_km": "17.2"},
        {"id_paquete": " A02 ", "peso_kg": "0.1", "distancia_km": "0"},
    ])
    entradas = inference.leer_csv(ruta)
    assert all(isinstance(entrada, Entrada) for entrada in entradas)
    assert [entrada.id_paquete for entrada in entradas] == ["Z09", "A02"]
    assert entradas[0].peso_kg == pytest.approx(3.27)
    assert entradas[0].distancia_km == pytest.approx(17.2)


def test_carga_csv_invalido():
    with pytest.raises(ValueError):
        inference.leer_csv(ROOT / "data/raw/paquetes_invalidos.csv")


@pytest.mark.parametrize("contenido", ["", "id_paquete,peso_kg,distancia_km\n"])
def test_csv_vacio(tmp_path, contenido):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text(contenido, encoding="utf-8")
    with pytest.raises(ValueError):
        inference.leer_csv(ruta)


# B4: vector en el orden indicado, redondeo, distancia e inmutabilidad.
@pytest.mark.parametrize("peso,distancia,esperado", [
    (2.36, 8, [2.4, 8.0]), (3.24, 17.27, [3.2, 17.27]),
    (0.16, 0, [0.2, 0.0]), (30, 200, [30.0, 200.0]),
])
def test_preprocesado(peso, distancia, esperado):
    entrada = valido(peso_kg=peso, distancia_km=distancia)
    antes = entrada.model_dump()
    vector = inference.preprocesar(entrada)
    assert isinstance(vector, list)
    assert vector == esperado
    assert entrada.model_dump() == antes


# B5: carga joblib, artefacto real y salida validada de una sola fila.
def test_carga_usa_joblib(tmp_path, monkeypatch):
    ruta = tmp_path / "modelo.joblib"
    modelo = ModeloPrueba()
    llamadas = []

    def cargar(ruta_recibida):
        llamadas.append(Path(ruta_recibida))
        return modelo

    monkeypatch.setattr(joblib, "load", cargar)
    assert inference.cargar_modelo(ruta) is modelo
    assert llamadas == [ruta]


def test_modelo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.cargar_modelo(tmp_path / "no_existe.joblib")


@pytest.mark.parametrize("categoria,probabilidades", [
    ("urgente", (0.23, 0.77)), ("normal", (0.81, 0.19)),
    ("normal", (0.5, 0.5)),
])
def test_predecir_matriz_y_confianza(categoria, probabilidades):
    entrada = valido()
    antes = entrada.model_dump()
    modelo = ModeloPrueba(categoria, probabilidades)
    salida = inference.predecir(entrada, modelo)
    assert isinstance(salida, Salida)
    assert salida.id_paquete == "P01"
    assert salida.categoria == categoria
    assert salida.confianza == pytest.approx(max(probabilidades))
    assert modelo.matrices_predict == [[[2.4, 8.0]]]
    assert modelo.matrices_proba == [[[2.4, 8.0]]]
    assert entrada.model_dump() == antes


@pytest.mark.parametrize("categoria,probabilidades", [
    ("otra", (0.2, 0.8)), ("normal", (-0.3, -0.2)),
    ("urgente", (0.2, 1.2)), ("normal", (float("nan"), float("nan"))),
    ("urgente", (0.2, float("inf"))),
])
def test_predecir_valida_salida(categoria, probabilidades):
    with pytest.raises(ValidationError):
        inference.predecir(valido(), ModeloPrueba(categoria, probabilidades))


def test_inferencia_real():
    modelo = inference.cargar_modelo(ROOT / "models/modelo.joblib")
    entrada = valido()
    salida = inference.predecir(entrada, modelo)
    matriz = [inference.preprocesar(entrada)]
    assert isinstance(salida, Salida)
    assert salida.id_paquete == entrada.id_paquete
    assert salida.categoria == modelo.predict(matriz)[0]
    assert salida.confianza == pytest.approx(max(modelo.predict_proba(matriz)[0]))


# B6: CSV con cabecera, orden y valores; ejecución sin resultados parciales.
def test_guardar_csv(tmp_path):
    ruta = tmp_path / "salida.csv"
    resultados = [
        Salida(id_paquete="Z09", categoria="urgente", confianza=0.77),
        Salida(id_paquete="A02", categoria="normal", confianza=0.81),
    ]
    inference.guardar_csv(resultados, ruta)
    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == ["id_paquete", "categoria", "confianza"]
        filas = list(lector)
    assert [fila["id_paquete"] for fila in filas] == ["Z09", "A02"]
    assert [fila["categoria"] for fila in filas] == ["urgente", "normal"]
    assert [float(fila["confianza"]) for fila in filas] == pytest.approx([0.77, 0.81])
    assert all(set(fila) == {"id_paquete", "categoria", "confianza"} for fila in filas)


def test_ejecucion_real(tmp_path):
    ruta = tmp_path / "resultado.csv"
    datos = ROOT / "data/raw/paquetes.csv"
    artefacto = ROOT / "models/modelo.joblib"
    inference.ejecutar(datos, artefacto, ruta)
    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == ["id_paquete", "categoria", "confianza"]
        filas = list(lector)
    entradas = inference.leer_csv(datos)
    modelo = inference.cargar_modelo(artefacto)
    assert len(filas) == len(entradas) == 6
    for entrada, fila in zip(entradas, filas):
        esperado = inference.predecir(entrada, modelo)
        assert fila["id_paquete"] == esperado.id_paquete
        assert fila["categoria"] == esperado.categoria
        assert float(fila["confianza"]) == pytest.approx(esperado.confianza)


def test_ejecutar_carga_modelo_una_vez(tmp_path, monkeypatch):
    llamadas = []
    modelo = ModeloPrueba()
    artefacto = tmp_path / "modelo.joblib"

    def cargar(ruta):
        llamadas.append(Path(ruta))
        return modelo

    monkeypatch.setattr(inference, "cargar_modelo", cargar)
    inference.ejecutar(ROOT / "data/raw/paquetes.csv", artefacto, tmp_path / "salida.csv")
    assert llamadas == [artefacto]
    assert len(modelo.matrices_predict) == 6
    assert len(modelo.matrices_proba) == 6


@pytest.mark.parametrize("existe", [False, True])
def test_no_salida_parcial_entrada_invalida(tmp_path, existe):
    ruta = tmp_path / "resultado.csv"
    if existe:
        ruta.write_bytes(b"original\n")
    with pytest.raises(ValueError):
        inference.ejecutar(ROOT / "data/raw/paquetes_invalidos.csv", ROOT / "models/modelo.joblib", ruta)
    if existe:
        assert ruta.read_bytes() == b"original\n"
    else:
        assert not ruta.exists()


@pytest.mark.parametrize("existe", [False, True])
def test_no_salida_parcial_prediccion_invalida(tmp_path, monkeypatch, existe):
    ruta = tmp_path / "resultado.csv"
    if existe:
        ruta.write_bytes(b"original\n")
    llamadas = []

    def predecir(entrada, modelo):
        llamadas.append(entrada.id_paquete)
        if len(llamadas) == 2:
            return Salida(id_paquete=entrada.id_paquete, categoria="incorrecta", confianza=0.8)
        return Salida(id_paquete=entrada.id_paquete, categoria="normal", confianza=0.8)

    monkeypatch.setattr(inference, "predecir", predecir)
    with pytest.raises(ValidationError):
        inference.ejecutar(ROOT / "data/raw/paquetes.csv", ROOT / "models/modelo.joblib", ruta)
    assert llamadas == ["P01", "P02"]
    if existe:
        assert ruta.read_bytes() == b"original\n"
    else:
        assert not ruta.exists()
