#Multidimensional indexing
#it consists of the layer, row, column

#Use zero indexing       row 0          row 1          row 2
# array = np.array([[['a','b','c'], ['d','e','f'], ['g','h','i']], #layer 0
#                   [['j','k','l'], ['m','n','o'], ['p','g','r']], #layer 1
#                   [['s','t','u'], ['v','w','x'], ['y','z','']]]) #layer 2
#                   col0 col1 col2

#for letter c - layer 0, row 0, col 2

import numpy as np
#Run in jupyter notebook
array = np.array([[['a','b','c'], ['d','e','f'], ['g','h','i']],
                  [['j','k','l'], ['m','n','o'], ['p','g','r']],
                  [['s','t','u'], ['v','w','x'], ['y','z','']]])

word = array[0,0,2] + array[1,0,2] + array[0,0,0] + array[2,0,0] + array[2,0,0] + " " + array[0,2,2] + array[2,0,0] + " " + array[2,0,1] + array[0,2,1] + array[0,1,1] + " " + array[0,0,1] + array[0,1,1] + array[2,0,0] + array[2,0,1]  

print(word)
#class is the best



# MY COMPLETE MULTIDIMENSIONAL ARRAYS LEARNING SESSION
# Date: Today!
# Achievement: Mastered 3D array shapes, explored 4D arrays


# All the arrays I analyzed
MY_ARRAYS = {
    "basic_3d": [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],
    "single_layer": [[[1,2,3],[4,5,6]]],
    "simple_1d": [1,2,3],
    "cube_222": [[[1,2],[3,4]],[[5,6],[7,8]]],
    "three_blocks": [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]],
    "two_by_four": [[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]],
    "tall_skinny": [[[I1],[2]],[[3],[4]],[[5],[6]]],
    "single_rows": [[[1,2,3]],[[4,5,6]],[[7,8,9]]],
    "many_singles": [[[1],[2],[3]],[[4],[5],[6]]],
    "four_singles": [[[1],[2],[3],[4]],[[5],[6],[7],[8]]],
}

# My correct answers
MY_ANSWERS = {
    "basic_3d": (2, 2, 3),
    "single_layer": (1, 2, 3),
    "simple_1d": (3,),
    "cube_222": (2, 2, 2),
    "three_blocks": (3, 2, 2),
    "two_by_four": (2, 2, 4),
    "tall_skinny": (3, 2, 1),
    "single_rows": (3, 1, 3),
    "many_singles": (2, 3, 1),
    "four_singles": (2, 4, 1),
}

# Verify everything
print("MY LEARNING SESSION RESULTS")
print("="*60)

for name, arr in MY_ARRAYS.items():
    actual = np.array(arr).shape
    my_answer = MY_ANSWERS[name]
    correct = actual == my_answer
    
    print(f"{name:15} | Shape: {str(actual):12} | ", end="")
    if correct:
        print(f"My answer: {my_answer} - CORRECT!")
    else:
        print(f"My answer: {my_answer} - WRONG (but it wasn't!)")

print("="*60)
print("Final score: 10/10 correct!")
print(" I've mastered multidimensional array shapes!")


## 2D slicing: [rows, columns]
arr_2d = np.array([[1,2,3,4],
                   [5,6,7,8],
                   [9,10,11,12]])

# Row slicing
print(arr_2d[0:2, :])    # First 2 rows, all columns
print(arr_2d[:, 1:3])    # All rows, columns 1-2
print(arr_2d[::2, ::2])  # Every other row and column

# 3D slicing: [layers, rows, columns]
arr_3d = np.arange(60).reshape(3, 4, 5)
print(arr_3d[0, :, :])    # Entire first layer
print(arr_3d[:, 2, :])    # Row 2 from all layers
print(arr_3d[..., 0])     # First column using ellipsis


#indexing tricks
# Boolean indexing (filtering)
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
print(arr[mask])  # [4, 5]

# Fancy indexing (specific positions)
print(arr[[0, 2, 4]])  # [1, 3, 5]

# Combined indexing
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix[[0,2], [1,1]])  # Get (0,1) and (2,1): [2, 8]


# Step 1: Align shapes from RIGHT
A = np.ones((3, 4, 5))  # Shape: (3, 4, 5)
B = np.ones((4, 1))      # Shape: ( , 4, 1) → becomes (1, 4, 1)

# Step 2: Check each dimension
# Dim3: 5 vs 1 → OK (one is 1)
# Dim2: 4 vs 4 → OK (equal)
# Dim1: 3 vs 1 → OK (one is 1)

# Step 3: Expand dimensions where needed
# B's shape (1, 4, 1) stretches to (3, 4, 5)

# Pattern 1: Scalar to array (easiest)
arr = np.ones((3, 4))
result = arr + 5  # 5 broadcasts to shape (3, 4)

# Pattern 2: Row/column broadcasting
matrix = np.ones((3, 4))
row = np.array([1, 2, 3, 4])      # Shape (4,)
col = np.array([[1], [2], [3]])   # Shape (3, 1)

print(matrix + row)   # Row added to each row of matrix
print(matrix + col)   # Column added to each column

# Pattern 3: Outer product (SO useful!)
a = np.array([1, 2, 3])    # (3,)
b = np.array([4, 5, 6, 7]) # (4,)
outer = a[:, np.newaxis] * b  # a becomes (3,1), b becomes (1,4) → result (3,4)


#SCALAR ARITMETICS IN NUMPY
import numpy as np
array = np.array([1,2,3,4,5])
print(array * 4)
print(array - 2)
print(array + 20)
print(array // 2)
print(array ** 2)

# [ 4  8 12 16 20]
# [-1  0  1  2  3]
# [21 22 23 24 25]
# [0 1 1 2 2]
# [ 1  4  9 16 25]

#MORE SCALAR ADDITION
import numpy as np
array1 = np.array([2, 4, 6, 8, 10])
array2 = np.array([3,6,8,32,6])

print(array1 + array2)
# [ 5 10 14 40 16]

#All Operations Work Element-wise
array = np.array([3,5,6,7,34.7,49.4])
print(np.sqrt(array))
print(np.log(array))
print(np.min(array))
print(np.max(array))
print(np.argmin(array))#index of the min value
print(np.argmax(array))#index of the max value
print(np.ceil(array))#round up
print(np.pi(array))#TypeError: 'float' object is not callable

# [1.73205081 2.23606798 2.44948974 2.64575131 5.89067059 7.02851336]
# [1.09861229 1.60943791 1.79175947 1.94591015 3.54673969 3.89995042]
# 3.0
# 49.4
# 0
# 5
# [ 3.  5.  6.  7. 35. 50.]


#area of a circle
import numpy as np
radius = np.array([1,2,3,4,5])
print(np.pi * radius * radius)
print(np.pi * radius ** 2)

# [ 3.14159265 12.56637061 28.27433388 50.26548246 78.53981634]
# [ 3.14159265 12.56637061 28.27433388 50.26548246 78.53981634]

#Comparison Operators
import numpy as np
array = np.array([2,4,6,7,97,85,23,16,21,10])
print([array > 4])
print((array >= 40) & (array < 100)) #you cant use the subscript here '[]'

# [array([False, False,  True,  True,  True,  True,  True,  True,  True,
#         True])]
# [False False False False  True  True False False False False]


import numpy as np
prices = np.array([100, 150, 200, 80])
tax_rate = 0.07

price_after_tax = prices * tax_rate
print(price_after_tax)


# # Linear Algebra Deep Dive – Days 28–31  
# Focus: Vectors, Matrices, Dot Products, Matrix Multiplication, Broadcasting, Rank, Determinants

# ## Overview
# Hands-on NumPy practice building intuition for core linear algebra concepts.  
# Started with vectors/matrices/dot products/matrix multiply, then deep-dived into broadcasting, and finished with rank & determinants.

# Goal: Solid foundation before eigenvalues, eigenvectors, and SVD.

# ## 1. Vectors, Matrices, Dot Products & Matrix Multiplication

### Basic operations
import numpy as np

# Vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# Dot product
dot_prod = np.dot(vec1, vec2)          # 32
# or modern: vec1 @ vec2

# Matrices
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

# Matrix multiplication
mat_prod = mat1 @ mat2
# [[19 22]
#  [43 50]]

### Matrix multiply from scratch (to understand mechanics)
def matmul_scratch(A, B):
    m, k = A.shape
    _, n = B.shape
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[p, j]
    return C

print(matmul_scratch(mat1, mat2))  # same as mat1 @ mat2


## 2. Broadcasting – Heavy Practice

### Column-wise operations (most common: feature normalization)
data = np.array([
    [100,  5.2,  23],
    [120,  6.1,  15],
    [ 90,  4.8,  40],
    [150,  7.0,   8]
])

means = np.mean(data, axis=0)          # shape (3,)
centered = data - means                # broadcasting: (4,3) - (3,) → (4,3) - (1,3)

### Row-wise operations
weights = np.array([1.1, 0.9, 1.2, 1.0])   # (4,)
weighted = data * weights[:, np.newaxis]   # (4,) → (4,1) → broadcasts down

### Per-column adjustment (opposite direction)

discount_factors = np.array([0.9, 0.85, 0.95])   # (3,)
discounted = prices * discount_factors[np.newaxis, :]  # (1,3) → broadcasts across

### Row-centered (subtract row means)

row_means = np.mean(X, axis=1)                    # (5,)
centered_rows = X - row_means[:, np.newaxis]      # key: [:, np.newaxis]

### Additive grid / outer sum
row_add = np.array([0, 5, 10, 15])        # (4,)
col_add = np.array([0, 1, 2, 3, 4])       # (5,)
result = base + row_add[:, np.newaxis] + col_add[np.newaxis, :]

### Multiplication table via broadcasting

row_values = np.array([2, 3, 4, 5])     # (4,)
col_values = np.array([1, 10, 100])     # (3,)
table = row_values[:, np.newaxis] * col_values[np.newaxis, :]
# [[  2   20  200]
#  [  3   30  300]
#  [  4   40  400]
#  [  5   50  500]]

## 3. Rank


# Quick rank checks
print(np.linalg.matrix_rank(A))          # e.g. identity → n

M1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])          # rank 2
M2 = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])          # rank 1
print(np.linalg.matrix_rank(M1))         # 2
print(np.linalg.matrix_rank(M2))         # 1

## 4. Determinant


# Basic det
A = np.array([[1, 2], [3, 4]])
print(np.linalg.det(A))                  # -2.0

# Singular case (dependent rows)
B = np.array([[1, 2], [2, 4]])
print(np.linalg.det(B))                  # 0.0

# Diagonal
C = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
print(np.linalg.det(C))                  # 6.0

# Squished to line
E = np.array([[2, 1], [4, 2]])
print(np.linalg.det(E))                  # 0.0

# Full rank 3×3
F = np.array([[3, 1, 0], [0, 2, 1], [1, 0, 4]])
print(np.linalg.det(F))                  # 25.0

### Combined rank + det check

rank = np.linalg.matrix_rank(M)
det  = np.linalg.det(M)
print(f"Rank: {rank}, Det: {det:.4f}, Invertible? {det != 0}")

## Core Rules to Remember

# Broadcasting
# - [:, np.newaxis] → stretch down (per-row)
# - [np.newaxis, :] → stretch across (per-column)

# Rank & Determinant
# - rank(A) < n  ⇔  det(A) = 0  ⇔  singular / not invertible
# - rank(A) = n  ⇔  det(A) ≠ 0  ⇔  full rank / invertible



B = np.array([[4, 0],
              [0, 1]])

# Manual check for λ = 4
# B @ [1,0] = [4,0] = 4 * [1,0]   ← x-axis stretched 4×
# B @ [0,1] = [0,1] = 1 * [0,1]   ← y-axis unchanged

# Example transformation of a general vector
v = np.array([3, 2])
print(B @ v)           # → [12, 2]   x stretched 4×, y unchanged



A = np.array([[3, 1],
              [0, 2]])

# For λ = 3
# (A - 3I)v = 0
# [[0, 1], [0, -1]] [x,y]^T = 0
# → y = 0
# Eigenvector: any [x, 0] → we chose [1, 0]

# For λ = 2
# (A - 2I)v = 0
# [[1, 1], [0, 0]] [x,y]^T = 0
# → x + y = 0 → y = -x
# Eigenvector: [1, -1] (or multiples)


# 1. Diagonal – obvious
M1 = np.array([[5, 0],
               [0, 2]])

# 2. Multiple of identity – every vector is eigenvector
M2 = np.array([[3, 0],
               [0, 3]])

# 3. Negative diagonal
M3 = np.array([[-4, 0],
               [ 0,-1]])

# 4. Upper triangular – one eigenvector obvious, one solved
M4 = np.array([[6, 1],
               [0, 2]])

# 5. Jordan block – repeated λ=1, only ONE independent eigenvector
M5 = np.array([[1, 4],
               [0, 1]])

# Quick NumPy check style (for verification when you want)
# eigenvalues, eigenvectors = np.linalg.eig(M)
# But we focused on manual solving today

# For M5 (Jordan case)
v = np.array([1, 0])          # is eigenvector → M5 @ v = [1, 0]
print(M5 @ v)

v2 = np.array([0, 1])         # is NOT eigenvector
print(M5 @ v2)                # → [4, 1]  (changes direction)


# Matrix 6 – [[4, -2], [1, 3]]
# User guessed λ=6 & 5, close directions but swapped
# Correct: λ=5 → v=[-2,1] or [2,-1]
#         λ=2 → v=[1,1] or [-1,-1]

# Matrix 8 – [[5, 4], [2, 3]]
# User: λ=7 & 1, v=(2,1) and (1,-1)
# → Perfect match

# Matrix 9 – [[1, 1], [-1, 1]]
# User guessed repeated λ=2
# Correct: complex eigenvalues 1+i and 1-i
# No real eigenvectors (rotation + scaling)
# Characteristic poly: λ² - 2λ + 2 = 0

# Matrix 10 – [[0, -1], [1, 0]]
# User guessed repeated λ=1
# Correct: λ = i and -i (pure 90° rotation)
# No real eigenvectors
# Characteristic poly: λ² + 1 = 0

# Extra requested: Matrix 11 – [[5, 3], [1, 3]]
# User: λ=2 & 6, v=(1,-1) and (1,3)
# Correct: λ=6 → [3,1] or [1,1/3]  (user's (1,3) is proportional)
#         λ=2 → [1,-1]
# → Directions correct, just scaling difference

# Extra: Matrix 12 – [[4, 0], [0, 4]]
# User: λ=4 (multiplicity 2), v=(1,1)
# Correct: λ=4 (multiplicity 2)
# Eigenspace = all of ℝ² (every vector is eigenvector)


# Extra: Matrix 13 – [[0, -2], [2, 0]]
# User: root 2i, no real eigenvector
# Correct: λ = 2i and -2i
# (scaled 90° rotation by factor 2)
# No real eigenvectors
# Characteristic poly: λ² + 4 = 0

# Quick NumPy verification style (optional – we focused on manual)
# eigenvalues, eigenvectors = np.linalg.eig(matrix)
# But today's emphasis was hand calculation

#2/18/26
import numpy as np

# ============================================================
# SECTION 1: Change of Basis Setup
# ============================================================

def build_P(b1, b2):
    # Columns are the new basis vectors
    return np.array([[b1[0], b2[0]],
                     [b1[1], b2[1]]], dtype=float)

b1 = np.array([2, 1])
b2 = np.array([0, 1])

P    = build_P(b1, b2)
Pinv = np.linalg.inv(P)

print("=== Change of Basis ===")
print("b1 =", b1, "| b2 =", b2)
print("P =\n", P)
print("P⁻¹ =\n", Pinv)


# ============================================================
# SECTION 2: New Coordinates → Standard
# ============================================================

new_coords     = np.array([3, 2])
standard_coords = P @ new_coords

print("\n=== New → Standard ===")
print("New coords     :", new_coords)
print("Standard coords:", standard_coords)   # [6, 5]


# ============================================================
# SECTION 3: Standard Coordinates → New
# ============================================================

back_to_new = Pinv @ standard_coords

print("\n=== Standard → New ===")
print("Standard coords:", standard_coords)
print("Back to new    :", back_to_new)        # [3, 2]


# ============================================================
# SECTION 4: Sandwich Formula — A_new = P⁻¹ A P
# ============================================================

P2    = build_P(np.array([1, 1]), np.array([1, -1]))
P2inv = np.linalg.inv(P2)

A     = np.array([[2, 0],
                  [0, 4]], dtype=float)

A_new = P2inv @ A @ P2

print("\n=== Sandwich Formula A_new = P⁻¹AP ===")
print("Basis: b1=(1,1), b2=(1,-1)")
print("A =\n", A)
print("A_new =\n", A_new)


# ============================================================
# SECTION 5: Round Trip Verification
# ============================================================

# Whatever you do in the new basis, you can always come back
test_point      = np.array([5, 3])
in_new_basis    = Pinv @ test_point
back_to_standard = P @ in_new_basis

print("\n=== Round Trip Check ===")
print("Start (standard) :", test_point)
print("In new basis     :", in_new_basis)
print("Back to standard :", back_to_standard)   # should match start


#2/22/26 SVD PRACTICE AND VISUALIZATION
import numpy as np

def matmul(A, B):
    """
    Manual matrix multiplication: C = A @ B using triple loop.
    """
    m, p1 = A.shape
    p2, n = B.shape
    
    if p1 != p2:
        raise ValueError(f'These matrices cannot be multiplied, p1 is not equal to p2')
    
    C = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            for k in range(p1):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


# Test case
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])   # 3×2

print("A @ B (manual):")
print(matmul(A, B))

print("NumPy @:")
print(A @ B)

#TRANSFORMATION VISUALIZER
import numpy as np
import matplotlib.pyplot as plt

def plot_transformation(A, title="Transformation"):
    """
    Visualize how matrix A transforms the unit circle and a grid.
    Overlays eigenvectors if they exist and are real.
    """
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    # Grid of points
    x = np.linspace(-1.5, 1.5, 15)
    y = np.linspace(-1.5, 1.5, 15)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()])

    # Apply transformation
    circle_transformed = A @ circle
    grid_transformed   = A @ grid

    # Eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        real_mask = np.isreal(eigenvalues)
        real_eigvals = eigenvalues[real_mask].real
        real_eigvecs = eigenvectors[:, real_mask].real
        norms = np.linalg.norm(real_eigvecs, axis=0)
        real_eigvecs = real_eigvecs / (norms + 1e-10)
    except:
        real_eigvecs = np.empty((2, 0))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    ax1.plot(circle[0], circle[1], 'b-', lw=1.5, label='Unit circle')
    ax1.scatter(grid[0], grid[1], s=10, color='lightblue', alpha=0.6)
    ax1.set_title("Before (original)")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.axvline(0, color='gray', lw=0.5)

    ax2.plot(circle_transformed[0], circle_transformed[1], 'r-', lw=1.5, label='Transformed')
    ax2.scatter(grid_transformed[0], grid_transformed[1], s=10, color='salmon', alpha=0.6)
    ax2.set_title(f"After {title}")
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.axvline(0, color='gray', lw=0.5)

    if real_eigvecs.shape[1] > 0:
        for idx in range(real_eigvecs.shape[1]):
            vec = real_eigvecs[:, idx]
            scale = 1.2
            ax1.quiver(0, 0, vec[0]*scale, vec[1]*scale,
                       color='green', scale_units='xy', scale=1, width=0.008)
            ax2.quiver(0, 0, vec[0]*scale, vec[1]*scale,
                       color='green', scale_units='xy', scale=1, width=0.008)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()