import torch
import torch.nn as nn

''' 3 DIFFERENT METHODS TO REMEMBER:
 - torch.save(arg, PATH) # can be model, tensor, or dictionary
 - torch.load(PATH)
 - torch.load_state_dict(arg)
'''

''' 2 FORMAS DIFERENTES DE AHORRAR
# 1) Lazy Way: Guardar todo el modelo
torch.save(modelo, RUTA)

# La clase de modelo debe definirse en algún lugar

model = torch.load(PATH)
model.eval()

# 2) Forma recomendada: guardar sólo el state_dict

torch.save(model.state_dict(), PATH)

# El modelo debe crearse de nuevo con parámetros

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
'''


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

# Entrenando el modelo

####################save all ######################################
for param in model.parameters():
    print(param)

# Guardar y cargar todo el modelo
FILE = "model.pth"
torch.save(model, FILE)

loaded_model = torch.load(FILE)
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)


############save only state dict #########################

# guardar solo el dictado de estado

FILE = "model.pth"
torch.save(model.state_dict(), FILE)

print(model.state_dict())
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE)) # it takes the loaded dictionary, not the path file itself
loaded_model.eval()

print(loaded_model.state_dict())


###########load checkpoint#####################

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

checkpoint = {
"epoch": 90,
"model_state": model.state_dict(),
"optim_state": optimizer.state_dict()
}
print(optimizer.state_dict())
FILE = "checkpoint.pth"
torch.save(checkpoint, FILE)

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']

model.eval()
# - or -
# model.train()

print(optimizer.state_dict())

# Recuerde que debe llamar a model.eval() para establecer capas de omisión y normalización por lotes 
# al modo de evaluación antes de ejecutar la inferencia. De lo contrario, se producirá 
# Resultados de inferencia inconsistentes. Si desea reanudar la formación, 
# Llame a model.train() para asegurarse de que estas capas están en modo de entrenamiento.

""" AHORRO EN GPU/CPU

# 1) Ahorre en GPU, cargue en CPU

device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# 2) Ahorre en GPU, cargue en GPU

device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Nota: Asegúrese de usar la función .to(torch.device('cuda')) 
# ¡En todas las entradas del modelo, también!

# 3) Ahorre en CPU, cargue en GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)

# Esto carga el modelo en un dispositivo GPU determinado. 
# A continuación, asegúrese de llamar a model.to(torch.device('cuda')) para convertir los tensores de parámetros del modelo en tensores CUDA
"""
