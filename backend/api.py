from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import os
import requests

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURAÇÕES ---
MODEL_PATH = './model/flower_resnet50.pth'
JSON_PATH = './model/cat_to_name.json'
# URL oficial usada nos cursos da Udacity para este dataset
JSON_URL = "https://raw.githubusercontent.com/udacity/content/master/deep-learning/image-classifier-project/cat_to_name.json"

device = torch.device("cpu")

# --- FUNÇÕES UTILITÁRIAS ---

def get_flower_names():
    """
    Carrega o mapeamento de nomes. Se o arquivo não existir, baixa automaticamente.
    """
    if not os.path.exists(JSON_PATH):
        print("Arquivo cat_to_name.json não encontrado. Baixando...")
        try:
            r = requests.get(JSON_URL)
            with open(JSON_PATH, 'wb') as f:
                f.write(r.content)
            print("Download concluído!")
        except Exception as e:
            print(f"Erro ao baixar nomes das flores: {e}")
            return None

    try:
        with open(JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler JSON: {e}")
        return None

def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Modelo ResNet50 carregado com sucesso!")
        return model
    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível carregar o modelo em {MODEL_PATH}")
        print(f"Detalhe: {e}")
        return None

# Carregar recursos na inicialização
model = load_model()
flower_names = get_flower_names()

# Transformação de imagem (Mesma usada no treino/validação)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "API Flower Identifier Online"}

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Modelo não carregado. Verifique se o arquivo .pth existe."}

    # Ler e preparar imagem
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Inferência
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Pegar top 3
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        # O modelo do PyTorch retorna índices de 0 a 101.
        # O arquivo cat_to_name.json usa chaves de "1" a "102".
        # Por isso somamos +1 ao índice do PyTorch.
        idx_pytorch = top3_catid[i].item()
        idx_json = str(idx_pytorch + 1)
        
        # Buscar nome ou usar fallback
        if flower_names:
            name = flower_names.get(idx_json, f"Espécie Desconhecida (ID {idx_json})")
        else:
            name = f"Flor ID {idx_pytorch}"
            
        score = top3_prob[i].item()
        
        results.append({
            "species": name.title(), # .title() deixa a primeira letra maiúscula
            "confidence": round(score * 100, 2)
        })

    return {"predictions": results}