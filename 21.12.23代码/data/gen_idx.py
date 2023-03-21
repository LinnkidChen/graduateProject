"""
this tool gens apa_idx.npy from apa.npz
"""

import numpy as np
from itertools import product

path = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/apcpa.npz"
f = np.load(path)
result = []
result = np.c_[f["row"], f["col"]]
print(result.shape)
print(f["row"].shape)

r_path = "/Users/tongchen/Library/Mobile Documents/com~apple~CloudDocs/毕业设计/graduateProject/21.12.23代码/data/dblp/apcpa_idx.npy"
np.save(r_path, result)
