# TERMINAL 1: Ejecutar Ollama
ollama serve

# TERMINAL 2: Instalar y ejecutar
pip install -r requirements.txt
python setup.py                    # Verificar entorno
python src/brain_ia.py             # Prueba rápida
python ejemplos_uso.py # En tu código:
from src.brain_ia import AsesorCiudadanoIA

asesor = AsesorCiudadanoIA()
asesor.indexar_leyes_locales()

denuncia = "Fui detenido sin orden judicial..."
defensa = asesor.generar_defensa_automatica(denuncia)

print(defensa['defensa']['fundamentos_legales'])            # 6 ejemplos completos
