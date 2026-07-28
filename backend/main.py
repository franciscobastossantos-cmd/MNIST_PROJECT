import os
import keras
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <--- 1. IMPORTA ISTO

app = FastAPI()

# --- 2. ADICIONA ESTA CONFIGURAÇÃO LOGO AQUI ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite que qualquer site fale com a API (ideal para testes)
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)
# -----------------------------------------------
# 1. Descobre onde este ficheiro está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Caminho corrigido: sobe da pasta 'backend' para a raiz, e entra em 'models'
model_path = os.path.join(BASE_DIR, "..", "models", "modelo_digitos.keras")

# 3. Carrega o modelo
model = keras.models.load_model(model_path)
# ----------------------------

@app.get("/")
def read_root():
    return {"status": "API pronta!"}

@app.post("/predict")
async def predict(data: dict):
    pixels = data.get("pixels", [])
    
    # Validação: Verifica se o tamanho está correto
    if len(pixels) != 784:
        return {"erro": f"Dados inválidos. Recebi {len(pixels)} pixels, mas o modelo precisa de 784."}
    
    pixel_data = np.array(pixels).reshape(1, 28, 28)
    prediction = model.predict(pixel_data)
    # 1. A classe (o número)
    predicted_class = int(np.argmax(prediction))

    # 2. A confiança (o valor máximo da probabilidade)
    confidence = float(np.max(prediction))

# Envia os dois no JSON
    return {"resultado": predicted_class, "confianca": confidence}