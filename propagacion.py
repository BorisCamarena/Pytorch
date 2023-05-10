import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

# Este es el parámetro que queremos optimizar -> requires_grad=True

w = torch.tensor(1.0, requires_grad=True)

# Pase hacia adelante para calcular la pérdida

y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

# paso hacia atrás para calcular el gradiente dLoss/dw

loss.backward()
print(w.grad)

# Actualizar pesos
# Siguiente pase hacia adelante y hacia atrás...

# Continuar optimizando:
# Actualizar pesos, esta operación no debe formar parte del gráfico computacional

with torch.no_grad():
    w -= 0.01 * w.grad

# No olvides poner a cero los gradientes

w.grad.zero_()

# Siguiente pase hacia adelante y hacia atrás...
