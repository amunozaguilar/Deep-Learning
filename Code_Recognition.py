# -*- coding: utf-8 -*-
"""
@author: Alex
"""
# Se recomienda el uso de Colab Google para ejecutar por etapas
# Habilitando el entorno de trabajo de Google Drive, en esta etapa nos solicitará una contraseña que debemos obtener entrando a nuestro correo
from google.colab import drive
drive.mount('/content/drive')

# Cargar librerias
import torch
import PIL
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn # Módulo están los modelos de red
import torch.nn.functional as F # Módulo donde están las funciones de activación

# Cargar imágenes
data_transform = transforms.Compose([
    transforms.Resize((128,128)),  
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomHorizontalFlip(p=0.6),                              
    transforms.RandomChoice([
      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                          scale=None, shear=None,
                          resample=False, fillcolor=0),
      transforms.ColorJitter(brightness=(1, 1.5),
                            contrast=(0.3, 2),
                            saturation=(0.1, 1),
                            hue=(-0.3, 0.3)),
      transforms.RandomRotation((-10, 10), resample=PIL.Image.BILINEAR)
    ]),
    transforms.ToTensor()
])

# Cargar base de datos desde https://www.kaggle.com/chetankv/dogs-cats-images?, copiada a GDrive
gatos_perros_dataset = datasets.ImageFolder(root='/content/drive/My Drive/...) # ... es la ruta donde se almacena el conjunto de datos
print(gatos_perros_dataset)

dataset_loader = torch.utils.data.DataLoader(gatos_perros_dataset,batch_size=32, shuffle=False, num_workers=2)
print(dataset_loader)

# Visualización de las imágenes 
plt.figure(num=None, figsize=(12, 8), dpi=80)

def imshow(img):
  np_img = img.numpy()
  plt.imshow(np.transpose(np_img,(1, 2, 0)))

# Obtener imagenes
data_iter = iter(dataset_loader)
images, labels = data_iter.next()

# Mostrar imagenes
imshow(torchvision.utils.make_grid(images))

# Separando la base de datos
# Transformaciones de la imagen, asi quedan mejor ajustadas
#data_transform = transforms.Compose([transforms.Resize((32,32)), transforms.RandomGrayscale(p=0.2),transforms.RandomVerticalFlip(p=0.2),transforms.RandomHorizontalFlip(p=0.2), transforms.ToTensor()])

data_transform = transforms.Compose([transforms.Resize((32,32)), transforms.RandomGrayscale(p=0.2),
                                    transforms.RandomVerticalFlip(p=0.6), transforms.RandomHorizontalFlip(p=0.6),                              
                                    transforms.RandomChoice([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=None, shear=None, resample=False, fillcolor=0), transforms.ColorJitter(brightness=(1, 1.5),
                                    contrast=(0.3, 2), saturation=(0.1, 1), hue=(-0.3, 0.3)), transforms.RandomRotation((-10, 10), 
                                    resample=PIL.Image.BILINEAR)]), transforms.ToTensor()])

data_transform_test = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

# Cargar las imágenes desde el acceso de las carpetas
gatos_perros_train = datasets.ImageFolder(root='.../training_set', transform=data_transform)
gatos_perros_valid = datasets.ImageFolder(root='.../valid_set',transform=data_transform_test)
gatos_perros_test = datasets.ImageFolder(root='.../test_set', transform=data_transform_test)

# Conjunto de entrenamiento
train_loader = torch.utils.data.DataLoader(gatos_perros_train, batch_size=32, shuffle=True, num_workers=2)

# Conjunto de validacion
valid_loader = torch.utils.data.DataLoader(gatos_perros_valid, batch_size=32, shuffle=False, num_workers=2)

# Conjunto de pruebas
test_loader = torch.utils.data.DataLoader(gatos_perros_test, batch_size=32, shuffle=False, num_workers=2)
                                           
                                           
# Revisión de las imágenes cargadas
plt.figure(num=None, figsize=(12, 8), dpi=80)

# Obtener imagenes
data_iter = iter(test_loader)
images, labels = data_iter.next()

# Mostrar imagenes
imshow(torchvision.utils.make_grid(images))

# Creación del modelo DL
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(720, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(x.shape[0], -1)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return x
    
model = CNN()
print(model)

# Configuración
# Hiperparámetros

num_epochs = 500
num_classes = 2
learning_rate = 0.001

# Configuración del dispositivo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Optimizador
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Etapa de entrenamiento
# Guardar las perdidas
train_losses = []
valid_losses = []

for epoch in range(1, num_epochs+1):
  ## Calcular el loss por cada iteracion
  train_loss = 0.0
  valid_loss = 0.0
  # Entrenar el model
  model.train()
  for data, target in train_loader:

    # Mover a GPU
    data = data.to(device)
    target = target.to(device)

    # Limpiar gradientes
    optimizer.zero_grad()
    # Prediccion
    output = model(data)
    # Medir el error
    loss = criterion(output, target)
    # Calcular gradientes
    loss.backward()
    # Actualizamos los pesos (Adam)
    optimizer.step()
    # Historia de la perdida
    train_loss += loss.item() * data.size(0)
  
  # Validar modelo (No calcula graientes, no actualizo pesos, no dropout)
  model.eval()
  for data, target in valid_loader:
     # Mover a GPU
    data = data.to(device)
    target = target.to(device) 

    # Prediccion para el conjunto de validacion
    output = model(data)

    loss = criterion(output, target)  

    valid_loss += loss.item() * data.size(0)

  # Calculo el promedio de Loss
  train_loss = train_loss/len(train_loader.sampler)
  valid_loss = valid_loss/len(valid_loader.sampler)

  train_losses.append(train_loss)
  valid_losses.append(valid_loss)

  # Imprimimos Loss (Pérdidas)
  print('Epoch: {} \tPerdida de entrenamiento: {:.6f} \tPerdida de Validacion: {:.6f}'.format(epoch, train_loss, valid_loss))
  
# Etapa de Pruebas 
# Probando el modelo
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
          
    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

# Guardar datos
torch.save(model.state_dict(), 'model.ckpt')

# Graficamos los resultados
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.plot(train_losses, label='Training loss', 'r')
plt.plot(valid_losses, label='Validation loss', 'b')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()
