# 1) Diseño del modelo (input, output)
# 2) Constructor de error - optimizacion.
# 3) Entrenamiento del loop
#       - Prediccion del error.
#       - Crearf el gradiente.
#       - Mejorar los coeficientes.

import torch
import torch.nn as nn

# Regresion Lineal.
# f = w * x 

# here : f = 2 * x

# Entrenamiento.

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# 1) Diseño del modelo: coeficientes optimizados con funcion.

# si requires_grad= True continua.

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Define el error y optimizacion.

learning_rate = 0.01
n_iters = 100 # iteraciones.

$ Define el error y la optimizacion.

loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr=learning_rate)

# entrena el loop.

for epoch in range(n_iters):
    y_predicted = forward(X)

    l = loss(Y, y_predicted)

 
    l.backward()


    optimizer.step()

 
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print('epoch ', epoch+1, ': w = ', w, ' loss = ', l)

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')
