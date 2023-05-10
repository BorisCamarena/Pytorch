import torch

# Todo en pytorch se basa en operaciones de Tensor.
# Un tensor puede tener diferentes dimensiones
# Así que puede ser 1D, 2D o incluso 3D y superior

# escalar, vector, matriz, tensor

# torch.empty(size): uninitiallized

x = torch.empty(1) # scalar
print(x)
x = torch.empty(3) # vector, 1D
print(x)
x = torch.empty(2,3) # matrix, 2D
print(x)
x = torch.empty(2,2,3) # tensor, 3 dimensions
#x = torch.empty(2,2,2,3) # tensor, 4 dimensions
print(x)

# torch.rand(size): random numbers [0, 1]
x = torch.rand(5, 3)
print(x)

# torch.zeros(size), fill with 0
# torch.ones(size), fill with 1
x = torch.zeros(5, 3)
print(x)

# tamaño del cheque

print(x.size())

# comprobar tipo de datos

print(x.dtype)

# especificar tipos, float32 predeterminado

x = torch.zeros(5, 3, dtype=torch.float16)
print(x)

print(x.dtype)

# construir a partir de datos

x = torch.tensor([5.5, 3])
print(x.size())

# requires_grad argumento
# Esto le dirá a pytorch que necesitará calcular los gradientes para este tensor
# más adelante en sus pasos de optimización
# es decir, esta es una variable en su modelo que desea optimizar

x = torch.tensor([5.5, 3], requires_grad=True)

# Operaciones

y = torch.rand(2, 2)
x = torch.rand(2, 2)

# adición elementwise

z = x + y
# torch.add(x,y)

# Además de lugar, todo con un guión bajo final es una operación en el lugar
# es decir, modificará la variable

# y.add_(x)

# resta

z = x - y
z = torch.sub(x, y)

# multiplicacion

z = x * y
z = torch.mul(x,y)

# division

z = x / y
z = torch.div(x,y)

# Rebanado

x = torch.rand(5,3)
print(x)
print(x[:, 0]) # all rows, column 0
print(x[1, :]) # row 1, all columns
print(x[1,1]) # element at 1, 1


# Obtenga el valor real si solo 1 elemento en su tensor

print(x[1,1].item())

# Reshape with torch.view()

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())

# Numpy
# Convertir un tensor de antorcha en una matriz NumPy y viceversa es muy fácil

a = torch.ones(5)
print(a)

# torch to numpy with .numpy()

b = a.numpy()
print(b)
print(type(b))

# Carful: Si el Tensor está en la CPU (no en la GPU),
# Ambos objetos compartirán la misma ubicación de memoria, por lo que cambiar uno
# también cambiará el otro

a.add_(1)
print(a)
print(b)

# numpy to torch with .from_numpy(x)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# De nuevo, tenga cuidado al modificar

a += 1
print(a)
print(b)

# por defecto todos los tensores se crean en la CPU,
# pero también puedes moverlos a la GPU (solo si está disponible)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    # z = z.numpy() # no es posible porque numpy no puede manejar tenores de GPU

    # Mover al GPU de nuevo.

    z.to("cpu")       # ``.to`` can also change dtype together!

    # z = z.numpy()
