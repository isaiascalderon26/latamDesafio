import fastapi
from challenge.model import DelayModel

app = fastapi.FastAPI()

# Crea una instancia de tu modelo para usar en la API.
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    try:
        # Realiza alguna validación de los datos de entrada si es necesario.
        # Luego, usa el modelo para hacer predicciones.
        # Supongo que los datos de entrada se proporcionan como un diccionario.
        features = pd.DataFrame([data])
        predicted_targets = model.predict(features)
        
        # Devuelve las predicciones como respuesta.
        return {
            "predictions": predicted_targets
        }
    except Exception as e:
        # Maneja cualquier excepción que pueda ocurrir.
        return {
            "error": str(e)
        }
