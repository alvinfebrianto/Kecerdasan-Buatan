# Alvin Febrianto
# 21091397031
# 2021 A / D4 Manajemen Informatika

# c. Multi Neuron Batch Input
#    i.   Input layer feature 10
#    ii.  Per batch nya 6 input
#    iii. Neuron 5

import numpy as np

# Inisialisasi variabel inputs
inputs = [[6.6, -7.2, 2.1, 6.8, 5.4, 1.9, 1.0, 6.9, 1.1, -4.3],
          [3.5, 8.6, -5.3, 6.8, 2.5, 2.0, 4.7, 1.3, -9.5, 7.0],
          [2.0, 7.9, 8.1, -2.3, 9.9, 6.3, 4.3, -8.8, 6.7, 8.2],
          [1.9, 4.7, 9.1, 3.3, -6.8, 5.2, -7.8, 5.0, 2.8, 3.7],
          [3.0, 4.8, 5.5, 8.6, -8.5, 7.0, -5.8, 5.0, 3.9, 3.6]]

# Inisialisasi variabel weights
weights = [[1.4, 2.0, 2.8, -4.2, 6.0, 2.4, -3.8, 1.2, 6.2, 5.1],
           [6.0, 6.6, 2.8, -8.2, 6.2, 1.6, 5.1, 3.6, -7.1, 4.0],
           [8.6, -2.3, 5.1, 7.4, 1.4, -2.4, 7.5, 5.7, 1.9, 8.4],
           [5.6, 4.0, -6.5, 8.2, 2.6, 1.8, -7.3, 7.8, 3.7, 5.9],
           [6.0, 3.0, 6.6, 1.3, -3.2, 7.1, 7.4, 9.4, 1.1, -7.6]]

# Inisialisasi variabel biases
biases = [2.0, 4, 0.8, 5.2, 3]

# Penghitungan output menggunakan rumus dot product vector
output = np.dot(inputs, np.array(weights).T) + biases
	
# Menampilkan output
print(output)