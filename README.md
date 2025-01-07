# pbCuda

This project demonstrates wrapping a CUDA module with Pybind11.

## Build and Install
```bash
mkdir build
cd build
cmake ..
make
```

## Test in Python
```python
import pbcuda
a = [1, 2, 3]
b = [4, 5, 6]
c = [0, 0, 0]
pbcuda.cuda_add(a, b, c)
print(c)  # Outputs: [5, 7, 9]
```
