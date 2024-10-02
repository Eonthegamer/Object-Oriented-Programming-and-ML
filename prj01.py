import numpy as np
import easynn as nn

# Create a numpy array of 10 rows and 5 columns.
# Set the element at row i and column j to be i+j.
def Q1():
    array = np.zeros((10, 5), dtype=int)

    for i in range(10):
        for j in range(5):
            array[i, j] = i + j

    return array

# Add two numpy arrays together.
def Q2(a, b):
    return a+b

# Multiply two 2D numpy arrays using matrix multiplication.
def Q3(a, b):
    result = np.dot(a, b)
    return result

# For each row of a 2D numpy array, find the column index
# with the maximum element. Return all these column indices.
def Q4(a):
    max_indices = []
    for row in a:
        max_index = np.argmax(row)
        max_indices.append(max_index)

    return max_indices

# Solve Ax = b.
def Q5(A, b):
    x = np.linalg.solve(A, b)
    return x


# Return an EasyNN expression for a+b.
def Q6():
    a = nn.Input("a")
    b = nn.Input("b")
    return a+b

# Return an EasyNN expression for a+b*c.
def Q7():
    a = nn.Input("a")
    b = nn.Input("b")
    c = nn.Input("c")
    d = a+b*c
    print(d.statements())
    return d

# Given A and b, return an EasyNN expression for Ax+b.
def Q8(A, b):
    A = nn.Const(A)
    b = nn.Const(b)
    x = nn.Input("x")
    f = A*x+b
    return f

# Given n, return an EasyNN expression for x**n.
def Q9(n):
    x = nn.Input("x")
    f = nn.Const(1)
    for i in range(n):
        f = x * f
    print(f.statements())
    return f

# Return an EasyNN expression to compute
# the element-wise absolute value |x|.
def Q10():
    x = nn.Input("x")
    r = nn.ReLU()
    f = r(x) + r(-x)
    return f
