import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Computabilidad de gradiente.

'''
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

# epoch = una pasada hacia adelante y hacia atrás de TODAS las muestras de entrenamiento
# batch_size = Número de muestras de entrenamiento utilizadas en una pasada hacia adelante/hacia atrás
# number of iterations = Número de pasadas, cada pasada (adelante + atrás) usando [batch_size] número de sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader puede hacer el cálculo por lotes por nosotros

# Implementar un conjunto de datos personalizado:
# implementar __init__ , __getitem__ , and __len__

class WineDataset(Dataset):

    def __init__(self):
        # Inicializar datos, descargar, etc.
        # leer con numpy o pandas
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # Aquí la primera columna es la etiqueta de clase, el resto son las características
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

# Admite la indexación de tal manera que el conjunto de datos[i] se puede usar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

# Podemos llamar a len(dataset) para devolver el tamaño
    def __len__(self):
        return self.n_samples

dataset = WineDataset()

# Obtenga la primera muestra y desempaque
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# Cargar todo el conjunto de datos con DataLoader
# shuffle: datos aleatorios, buenos para el entrenamiento
# num_workers: carga más rápida con múltiples subprocesos
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)


# Convertir a un iterador y mirar una muestra aleatoria
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        # Correr el proceso de entrenamiento.
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

# Algunos conjuntos de datos famosos están disponibles en Torchvision.datasets
# por ejemplo, MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)


dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)
