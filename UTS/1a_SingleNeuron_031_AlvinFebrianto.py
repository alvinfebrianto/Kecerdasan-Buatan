# Alvin Febrianto
# 21091397031
# 2021 A / D4 Manajemen Informatika

# a. Single Neuron
#    i.  Input layer feature 10
#    ii. Neuron 1

import numpy as np

# Inisialisasi variabel inputs
inputs = [0.8, 4, 7, -3.2, 1.3, 3, -8, -2.0, 4.5, 2.4]

# Inisialisasi variabel weights
weights	= [-3.6, 3.2, 6, -1.8, 2.8, -7, 5.7, 3.4, 3.1, 4.2]

# Inisialisasi variabel bias
bias = 6

# Penghitungan output menggunakan rumus dot product vector (inputs*weights)+bias
output = np.dot(weights, inputs) + bias

# Menampilkan output
print(output)