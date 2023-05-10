
import torch

# El paquete autograd proporciona diferenciación automática 
# para todas las operaciones en tensores

# requires_grad = True -> rastrea todas las operaciones en el tensor.

x = torch.randn(3, requires_grad=True)
y = x + 2

# y se creó como resultado de una operación, por lo que tiene un atributo grad_fn.
# grad_fn: hace referencia a una función que ha creado el tensor

print(x) # created by the user -> grad_fn is None
print(y)
print(y.grad_fn)

# Hacer más operaciones en y

z = y * y * 3
print(z)
z = z.mean()
print(z)

# Calculemos los gradientes con retropropagación
# Cuando terminemos nuestro cálculo podemos llamar a .backward() y tener todos los gradientes calculados automáticamente.
# El degradado de este tensor se acumulará en el atributo .grad.
# Es la derivada parcial de la función w.r.t. el tensor

z.backward()
print(x.grad) # dz/dx

# En términos generales, torch.autograd es un motor para calcular productos vectoriales-jacobianos
# Calcula derivadas parciales mientras aplica la regla de la cadena

# -------------
# Modelo con salida no escalar:
# Si un tensor no es escalar (más de 1 elemento), necesitamos especificar argumentos para backward() 
# Especifique un argumento de degradado que sea un tensor de forma coincidente.
# necesario para el producto vectorial-jacobiano

x = torch.randn(3, requires_grad=True)

y = x * 2
for _ in range(10):
    y = y * 2

print(y)
print(y.shape)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)

# -------------
# Detener un tensor del historial de seguimiento:
# Por ejemplo durante nuestro bucle de entrenamiento cuando queremos actualizar nuestras pesas
# Entonces esta operación de actualización no debería formar parte del cálculo de degradado
# - x.requires_grad_(Falso)
# - x.detach()
# - envolver 'con torch.no_grad():'

# .requires_grad_(...) cambia un indicador existente en su lugar.

a = torch.randn(2, 2)
print(a.requires_grad)
b = ((a * 3) / (a - 1))
print(b.grad_fn)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# .detach(): obtiene un nuevo tensor con el mismo contenido pero sin cálculo de gradiente:

a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)

# wrap in 'with torch.no_grad():'

a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

# -------------
# backward() acumula el degradado de este tensor en el atributo .grad.
# !!! Debemos tener cuidado durante la optimización !!!
# ¡Use .zero_() para vaciar los degradados antes de un nuevo paso de optimización!

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):

    # solo un ejemplo ficticio

    model_output = (weights*3).sum()
    model_output.backward()
    
    print(weights.grad)

    # optimizar la modelo, es decir, ajustar los pesos...

    with torch.no_grad():
        weights -= 0.1 * weights.grad

    # Esto es importante! Afecta a los pesos finales y la producción

    weights.grad.zero_()

print(weights)
print(model_output)

# Optimizer has zero_grad() method
# optimizer = torch.optim.SGD([weights], lr=0.1)
# During training:
# optimizer.step()
# optimizer.zero_grad()
