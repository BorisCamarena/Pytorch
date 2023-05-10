import torch

# ---------------------------------------------------------
# En pytorch todo esta basado en operaciones tensoriales.
# ---------------------------------------------------------
# Un tensor vive en Rn x Rm x ...  
# Escalar vacio.



# torch.empty(size): uninitiallized
x = torch.empty(1) # scalar
print(x)
# Vector en R3.
x = torch.empty(3) # vector, 
print(x)
# Tensor en R3xR3
x = torch.empty(2,3) 
print(x)
# Tensor en R2XR3xR2
x = torch.empty(2,2,3) # tensor, 7 dimensions
#x = torch.empty(2,2,2,3) # tensor, 4 dimensions
print(x)

# torch.rand(size): random numbers [0, 1]
# tensor de numero aleatorios, en R3XR5
# Tensor de R5XR3 relleno con ceros.

x = torch.rand(5, 3)
print(x)

# torch.zeros(size), llenar con  0
# torch.ones(size), llenar con1 1
x = torch.zeros(5, 3)
print(x)
# Checar tamaño del tensor.
# lista con dimensiones.
# check size
print(x.size())
# Checar tipo de datos float32.
# check data type
print(x.dtype)
# Especificando tipos de datos.
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)

# check type
print(x.dtype)
# Construir vector con datosl.
x = torch.tensor([5.5, 3])
print(x.size())

# requires_grad argument
# This will tell pytorch that it will need to calculate the gradients for this tensor
# later in your optimization steps
# i.e. this is a variable in your model that you want to optimize

# Vector utilizable (requiere gradiente).

x = torch.tensor([5.5, 3], requires_grad=True)

# Suma de tensores componente a componente.
y = torch.rand(2, 2)
x = torch.rand(2, 2)

# elementwise addition
z = x + y
# torch.add(x,y)

# in place addition, everythin with a trailing underscore is an inplace operation
# i.e. it will modify the variable
# y.add_(x)

# Resta de tensores.
z = x - y
z = torch.sub(x, y)

# Multiplicacion de tensores.
z = x * y
z = torch.mul(x,y)

# Division de tensores.
z = x / y
z = torch.div(x,y)

# Rebanadas de tensores.

x = torch.rand(5,3)
print(x)
print(x[:, 0]) # Todosm los renglones, columna 0
print(x[1, :]) # Renglon 1, todas las columnas.
print(x[1,1]) # Elemento en 1, 1

# In dice del elemento 
print(x[1,1].item())

# Reshape with torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())

# Numpy
# Converting a Torch Tensor to a NumPy array and vice versa is very easy
a = torch.ones(5)
print(a)

# torch to numpy with .numpy()
b = a.numpy()
print(b)
print(type(b))

# Carful: If the Tensor is on the CPU (not the GPU),
# both objects will share the same memory location, so changing one
# will also change the other

# Suma 1 a todas las entradas.

a.add_(1)
print(a)
print(b)

# De numpy a torch.

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# Le suma 1 a todas las entradas de a.
a += 1
print(a)
print(b)


# De CPU a GPU si hay CUDA.

# by default all tensors are created on the CPU,
# but you can also move them to the GPU (only if it's available )
if torch.cuda.is_available():
    device = torch.device("cuda")          # La tarjeta de video con CUDA.
    y = torch.ones_like(x, device=device)  # Crear tensor en GPU.
    x = x.to(device)                       # Copíar a GPU ``.to("cuda")``
    z = x + y
    # ----------------------------------------------------------------------------
    # z = z.numpy() # not possible because numpy cannot handle GPU tenors
    # Numpy maneja tensores en el GPU.
    # De vuelta al CPU



    z.to("cpu")       # ``.to`` can also change dtype together!
    z = z.numpy()
