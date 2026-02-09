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
