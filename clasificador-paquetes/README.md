#
Instalar:
Con git clone te clonas el repo
git clone

Como el fichero tiene pyproject y uv.lock con uv sync se te instala todas las dependencias y versiones necesarias
uv sync

Para ejecutarlo:
uv run python -m clasificador_paquetes.inference
uv run pytest