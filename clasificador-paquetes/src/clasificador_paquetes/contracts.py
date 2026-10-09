from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Entrada(BaseModel):
    #Los números llegan como texto desde el CSV: deben aceptarse textos numéricos y rechazarse textos no numéricos. En ambas se rechazan campos extra.
    '''Entrada
        id_paquete: texto. Elimina los espacios al principio y al final; después no puede estar vacío.
        peso_kg: número finito, mayor que 0 y menor o igual que 30.
        distancia_km: número finito, entre 0 y 200, ambos incluidos.
        Salida

        id_paquete: texto no vacío.
        categoria: solo normal o urgente.
        confianza: número finito entre 0 y 1, ambos incluidos.'''
    model_config = ConfigDict(extra="forbid")
    id_paquete: str = Field(min_length=1)
    peso_kg: float = Field(gt=0, le=30)
    distancia_km: float = Field(ge=0, le=200)
    @field_validator("id_paquete", mode="before")
    @classmethod
    def space_remove(cls, value):
        value = value.strip(" ")
        return value
   

class Salida(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_paquete: str = Field(min_length=1)
    categoria: Literal["normal", "urgente"]
    confianza: float = Field(ge=0, le=1)

