# pbCuda

This project demonstrates wrapping a CUDA module with Pybind11.

## Build and Install
```bash
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make && make install 
```
Change ``native`` to ``nn`` for corresponding nvidia GPUS, e.g., ``70`` for V100, ``80`` for A100. 

Or use ``pip`` method
```bash
python3 -m pip install . --no-deps --ignore-installed -vv
```


## Test in Python
```python
import pbcuda
import numpy as np 
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([4, 5, 6], dtype=np.int32)

# Perform CUDA addition
c = pbcuda.cuda_add(a, b)

print(c)  # Outputs: [5, 7, 9]
```
