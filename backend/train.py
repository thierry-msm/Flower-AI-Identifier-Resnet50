import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Configurações
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = './dataset'
MODEL_SAVE_PATH = './model/flower_resnet50.pth'

def train_model():
    # 1. Configurar Transformações (Data Augmentation para melhorar a generalização)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Baixar e Carregar Dataset Oxford Flowers 102
    print("Baixando e preparando dataset...")
    os.makedirs('./model', exist_ok=True)
    
    # O split 'train' do Oxford102 é pequeno, então usaremos ele para treino
    # e 'test' para validação neste exemplo simplificado.
    train_dataset = datasets.Flowers102(root=DATA_DIR, split='train', download=True, transform=data_transforms['train'])
    val_dataset = datasets.Flowers102(root=DATA_DIR, split='test', download=True, transform=data_transforms['test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Treinando usando: {device}")

    # 3. Carregar Modelo Pré-treinado (ResNet50)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Congelar as camadas iniciais (opcional, mas bom para datasets pequenos)
    for param in model.parameters():
        param.requires_grad = False

    # 4. Substituir a última camada (Fully Connected) para 102 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102) # Oxford Flowers tem 102 classes

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # 5. Loop de Treinamento
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 6. Salvar o Modelo
    print(f"Salvando modelo em {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Treinamento concluído!")

if __name__ == '__main__':
    train_model()