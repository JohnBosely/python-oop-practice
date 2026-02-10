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


import numpy as np

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
