import pbcuda
import numpy as np

def test_cuda_add():
    # Prepare test data
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)

    # Perform CUDA addition
    c = pbcuda.cuda_add(a, b)

    # Check results
    expected = np.array([5, 7, 9], dtype=np.float32)
    assert np.array_equal(c, expected), f"Expected {expected}, got {c}"

if __name__ == "__main__":
    test_cuda_add()
    print("All tests passed!")
