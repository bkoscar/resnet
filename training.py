from PIL import Image
import matplotlib.pyplot as plt
import os 
import torch
from data_processing import DataIter
from model import ResNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.multiprocessing as mp


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
torch.manual_seed(43)
train_folder = '\\Users\\oscar\\OneDrive\\Documents\\pytorch\\CNNs\\data\\archive\\train'
test_folder = '\\Users\\oscar\\OneDrive\\Documents\\pytorch\\CNNs\\data\\archive\\test'
a = DataIter(train_folder, test_folder, batch_size = 32)
data_iter_train = a.dataloader()[0]

model = ResNet(num_classes=13).to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)



if __name__ == "__main__":
    mp.freeze_support()
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        # Utilizar tqdm para imprimir una barra de progreso por cada época
        with tqdm(data_iter_train, unit="batch") as t:
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Actualizar la descripción de la barra de progreso con el valor de la pérdida
                t.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
