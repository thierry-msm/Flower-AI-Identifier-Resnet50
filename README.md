# FlowerAI Identifier

Este projeto é uma aplicação **Full Stack de Inteligência Artificial**
desenvolvida para a classificação e identificação automática de espécies
de flores. O sistema distingue **102 espécies diferentes** utilizando
uma **Rede Neural Convolucional (ResNet50)** baseada em **Transfer
Learning**.

O projeto é dividido em dois componentes principais:

-   **Backend (API):** Desenvolvido em Python com FastAPI e PyTorch,
    responsável pelo processamento da imagem e inferência do modelo.\
-   **Frontend (Interface):** Desenvolvido em React com Vite e Tailwind
    CSS, oferecendo uma interface moderna e responsiva.

------------------------------------------------------------------------

## 🛠 Tecnologias Utilizadas

### Backend & IA

-   **Linguagem:** Python 3.8+\
-   **Framework Web:** FastAPI (com Uvicorn)\
-   **Machine Learning:** PyTorch, Torchvision\
-   **Modelo:** ResNet50 (Transfer Learning)\
-   **Processamento de Imagem:** Pillow (PIL)

### Frontend

-   **Framework:** React (Vite)\
-   **Estilização:** Tailwind CSS\
-   **Ícones:** Lucide React\
-   **Linguagem:** JavaScript (ES6+)

------------------------------------------------------------------------

## 📋 Pré-requisitos

Para rodar o projeto localmente, você precisará de:

-   Python 3.8 ou superior\
-   Node.js 20 ou superior\
-   Git

------------------------------------------------------------------------

## 🚀 Como Rodar o Projeto

Recomenda-se abrir **dois terminais**: um para o Backend e outro para o
Frontend.

------------------------------------------------------------------------

### **Passo 1: Configuração do Backend**

Acesse a pasta do backend:

``` bash
cd backend
```

Crie um ambiente virtual:

``` bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências:

``` bash
pip install -r requirements.txt
```

### Treinamento/Modelo

Coloque o arquivo do modelo treinado `flower_resnet50.pth` dentro de:

    backend/model/

#### Opção A -- Treinar localmente

``` bash
python train.py
```

#### Opção B -- Modelo já treinado

Baixe o `.pth` treinado (ex.: via Google Colab) e coloque em:

    backend/model/flower_resnet50.pth

Inicie o servidor da API:

``` bash
uvicorn api:app --reload
```

A API estará disponível em: **http://127.0.0.1:8000**

------------------------------------------------------------------------

### **Passo 2: Configuração do Frontend**

Abra um novo terminal e acesse:

``` bash
cd frontend
```

Instale as dependências:

``` bash
npm install
```

Execute o app:

``` bash
npm run dev
```

Acesse no navegador: **http://localhost:5173**

------------------------------------------------------------------------

## 🧪 Como Usar

1.  Certifique-se de que o Backend está rodando (porta 8000).\
2.  Abra a interface web no navegador.\
3.  Envie ou arraste uma imagem de flor (.jpg ou .png).\
4.  Clique em **"Identificar Espécie"**.\
5.  O sistema exibirá as **3 espécies mais prováveis** com suas
    porcentagens de confiança.

------------------------------------------------------------------------

## 📂 Estrutura de Pastas

``` text
flower_model/
├── backend/
│   ├── model/           # Arquivos do modelo (.pth e .json)
│   ├── api.py           # Código da API FastAPI
│   ├── train.py         # Script de treinamento da IA
│   └── requirements.txt # Dependências Python
│
└── frontend/
    ├── src/             # Código fonte React
    ├── public/          # Assets estáticos
    ├── index.html       # HTML principal
    └── vite.config.js   # Configuração do Vite
```
