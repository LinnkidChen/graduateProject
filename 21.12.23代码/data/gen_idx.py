"""
this tool gens apa_idx.npy from apa.npz
"""

import numpy as np
from itertools import product

path = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/aptpa.npz"
f = np.load(path)
result = []
for i in range(f["col"].size):
    result += [[i, j]]
result = np.array(result)
print(result.shape)

r_path = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/aptpa_idx.npy"
np.save(r_path, result)
