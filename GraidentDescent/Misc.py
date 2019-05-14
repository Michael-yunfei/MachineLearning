# Miscellaneous code
# @ Michael

import numpy as np

store_vector = np.empty([0, 1])

i = 0
while i <= 100:
    store_vector = np.append(store_vector, np.array(i).reshape([1, 1]),
                             axis=0)
    i += 1


store_vector
