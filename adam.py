import numpy as np

# Data contoh
X = np.array([
    [2104, 3],
    [1600, 3],
    [2400, 3],
    [1416, 2],
    [3000, 4]
])

y = np.array([399900, 329900, 369000, 232000, 539900])

# Jumlah epoch
num_epochs = 1000

# Hyperparameters
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Inisialisasi parameter
w1, w2, b = 0.0, 0.0, 0.0
m_w1, v_w1 = 0.0, 0.0
m_w2, v_w2 = 0.0, 0.0
m_b, v_b = 0.0, 0.0

t = 0

# Implementasi Adam
for epoch in range(num_epochs):
    for i in range(len(X)):
        t += 1
        x1, x2 = X[i]
        y_true = y[i]

        # Prediksi
        y_pred = w1 * x1 + w2 * x2 + b
        print(y_pred)

        # Hitung gradien
        grad_w1 = -(2 / len(X)) * x1 * (y_true - y_pred)
        grad_w2 = -(2 / len(X)) * x2 * (y_true - y_pred)
        grad_b = -(2 / len(X)) * (y_true - y_pred)
        print(f"{grad_w1} {grad_w2} {grad_b}")

        # Update momen pertama dan kedua
        m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1
        v_w1 = beta2 * v_w1 + (1 - beta2) * (grad_w1 ** 2)
       
        m_w2 = beta1 * m_w2 + (1 - beta1) * grad_w2
        v_w2 = beta2 * v_w2 + (1 - beta2) * (grad_w2 ** 2)
        
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

        # Koreksi bias
        m_w1_hat = m_w1 / (1 - beta1 ** t)
        v_w1_hat = v_w1 / (1 - beta2 ** t)

        m_w2_hat = m_w2 / (1 - beta1 ** t)
        v_w2_hat = v_w2 / (1 - beta2 ** t)

        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        # Update parameter
        w1 -= alpha * m_w1_hat / (np.sqrt(v_w1_hat) + epsilon)
        w2 -= alpha * m_w2_hat / (np.sqrt(v_w2_hat) + epsilon)
        b -= alpha * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

# Parameter akhir setelah pelatihan
print(f"w1: {w1}, w2: {w2}, b: {b}")
