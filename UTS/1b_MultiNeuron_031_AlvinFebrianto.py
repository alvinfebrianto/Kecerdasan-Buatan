# Alvin Febrianto
# 21091397031
# 2021 A / D4 Manajemen Informatika

# b. Multi Neuron
#    i.  Input layer feature 10
#    ii. Neuron 5

import numpy as np

# Inisialisasi variabel inputs
inputs = [-8, 7.1, -5, 4.2, 5.3, 2.0, 7.9, 2.4, 8.4, -4]

# Inisialisasi variabel weights
weights = [[1.4, 2.0, 2.8, -4.2, 6.0, 2.4, -3.8, 1.2, 6.2, 5.1],
           [6.0, 6.6, 2.8, -8.2, 6.2, 1.6, 5.1, 3.6, -7.1, 4.0],
           [8.6, -2.3, 5.1, 7.4, 1.4, -2.4, 7.5, 5.7, 1.9, 8.4],
           [5.6, 4.0, -6.5, 8.2, 2.6, 1.8, -7.3, 7.8, 3.7, 5.9],
           [6.0, 3.0, 6.6, 1.3, -3.2, 7.1, 7.4, 9.4, 1.1, -7.6]]

# Inisialisasi variabel biases
biases = [2.0, 4, 0.8, 5.2, 3]

# Penghitungan output menggunakan rumus dot product vector (inputs*weights)+biases
output = np.dot(weights, inputs) + biases

# Menampilkan output
print(output)