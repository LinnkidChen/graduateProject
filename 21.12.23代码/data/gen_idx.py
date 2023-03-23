"""
this tool gens apa_idx.npy from apa.npz
"""

import numpy as np
from itertools import product

path = "/root/graduateProject/21.12.23代码/data/freebase/mwm.npz"
f = np.load(path)
result = []
result = np.c_[f["row"], f["col"]]
print(result.shape)
print(f["row"].shape)

r_path = "/root/graduateProject/21.12.23代码/data/freebase/mwm_idx.npy"
np.save(r_path, result)
