import numpy as np
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]


file1 = np.load(file1)
file2 = np.load(file2)

if np.allclose(file1, file2, rtol=1e-4, atol=1e-4):
    print("Files are identical upto 4 decimals")
else:
    file1 = file1.flatten()
    file2 = file2.flatten()

    print("Files are different:", file1, file2)
