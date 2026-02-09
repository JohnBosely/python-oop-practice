PYTHON OOP PRACTICE
This is a collection of projects i did on Object Oriented Programming, i completed them all in a week, the learning process, reading and everything
Project Structure
python-oop-practice/
│
├── README.md
├── encapsulation/
│   ├── bank_account.py
│   ├── temperature.py
│   ├── student.py
│   ├── user.py
│   ├── shopping_cart.py
│   ├── fuel_tank.py
│   ├── music_player.py
│   └── todo_list.py
│
├── data_structures/
│   ├── student_management.py
│   └── inventory_tracker.py
│
└── inheritance/
    ├── vehicle.py
    ├── employee.py
    └── shape.py


TOPICS COVERED
Encapsulation - Private attributes, properties, getters/setters
Data Structures - Lists, dictionaries, sets, tuples
Inheritance - Parent/child classes, super()

HOW TO RUN
Ech file can basically be run independently, you dont need any fancy thing to run it, just use bash.

python encapsulation/bank_account.py
python data_structures/inventory_tracker.py
python inheritance/vehicle.py

KEY CONCEPTS I LEARNED
OOP was quite easy for me to understand foe some weird reason, genuinely dont know how i understood it so well. I'll probably say its because of how fun i made it, i did a few projects (not here though) using game scenarios, it involved the characters health, stats, and other stuff. Gamification of topics really helped me, brocode and Mosh were so so important also.

This is a summary of what i learnt
Encapsulation
 Using _ prefix for private attributes
 Creating @property decorators
 Implementing setters with validation
 Data protection

 Data Structures
 Nested dictionaries
 Dictionary methods (.items(), .values(), .keys())
 List comprehensions
 CRUD operations

 Inheritance
 Parent/child relationships
 Using super()
 Method overriding
 Code reusability


Projects

Encapsulation (8 projects)
 Bank Account, Temperature Converter, Student Grade Manager
 User Authentication, Shopping Cart, Fuel Tank
 Music Player, Todo List

Data Structures (2 projects)
 Student Management System (full CRUD)
 Inventory Tracker (warehouse management)

Inheritance (3 exercises)
 Vehicle Hierarchy, Employee System, Shape Calculator

Next Steps
 Abstraction
 Polymorphism
 Final project combining all concepts

For the next steps i'll be using the same gamified method to learn, i think its really important as it helps me think and understand the structure of lists, dictionaries, and other stuff

STRUGGLES
To be honest they were mostly down to memorizing syntax, which i really try to stop doing, so i learnt a very efficinet method to understabd and breakdown methods
I got it from "PYTHON 3 OBJECT ORIENTED PROGRAMMING 2ND EDITION BY DUSTY PHILLIPS" 

1. START WITH THE PROBLEM NOT THE CODE
2. IDENTIFY OBVIOUS OBJECTS FROM DOMAIN KNOWLEDGE
3. IDENTIFY ATTRIBUTES OR DATA OF THOSE OBJECTS
4. RECOGNIZE RESPONSIBILITIES
5. QUESTION INITIAL ASSUMPTIONS(THUS BUTCHERED ME A LOT COS WHEN I THOUGHT DEEPLY ABOUT SOME OBJECTS OR EVEN ATTRIBUTES, SOME FELT WAY TO OBVIOUS AND INSTEAD OF THINKING EVEN MORE OR WRITING THEM DOWN, I STARTED CODING IMMEDIATELY. IT MADE ME REALLY FRUSTRATED AND ALSO DOUBT MYSELF BUT I HELPED ME EVENTUALLY)
6. REFINE THE MODEL (VERY IMPORTANT)

THIS IS BASICALLY ALL YOU NEED TO BE HONEST.


ABSTRACTION AND POLYMORPHISM
Just learnt abstraction, its the hiding of complexity. This concept was really vague cos i was wondering why there would be methods(abstract methods) that didnt have built in functions, then this led to polymorphism which means "many forms". 
Abstraction involves using the:
from abc import ABC, abstractmethod

We imported this because python doesnt necessarily support abstraction so the ABC(abstract base classes) was created.

STRUGGLES
With Abstraction:
"Why abstract when I can just write normal classes?" i asked myself this question a lot.

Struggle to see practical benefits initially

Solution: Work on larger projects where abstraction prevents code duplication

Abstract vs Concrete class confusion

When to make a class abstract vs regular

Remember: Abstract classes = templates, Concrete = usable objects (learn this from a friend)

Python's flexibility backfires

No strict enforcement like Java/C#

Easy to bypass abstraction, leading to poorly structured code

With Polymorphism:
Recognizing when to create polymorphic structures vs simple conditionals(definitely my biggest issue)

TIPS THAT HELPED ME
1. Build small projects, like the shape project, then increase the difficulty.
2. Watch BroCode (he's so good)
3. Write down the algorith before coding, this helped me alot honestly. I had a lot of "aha" moments while doing this



On the 9th of February 2026
NUMPY (I used Bro Code - the 1 hr video https://youtu.be/VXU4LSAQDSc?si=0caYD6uQerSpmSH_)

Today's Learning Journey: Conquering Multidimensional Arrays!

The Breakthrough Moment
Today was the day it finally clicked. After wrestling with confusing bracket patterns and getting lost in nested lists, I finally understand multidimensional arrays in NumPy.

What I Mastered

The Core Insight
I discovered that every extra pair of brackets adds a dimension, and the secret is to read shapes "from the outside in" (for some reason this took time):
- [1,2,3] → 1D, shape: (3,)
- [[1,2],[3,4]] → 2D, shape: (2, 2)
- [[[1,2],[3,4]]] → 3D, shape: (1, 2, 2)

My Foolproof Method
I developed a personal system for decoding even the messiest arrays:
1. Count consecutive "[" at the start → That's the number of dimensions!
2. Read commas at each level → Number of elements at that dimension
3. Work from outermost to innermost → Build the shape step by step

Key Realizations
- The difference between [[1,2,3]] (shape: (1, 3)) and [1,2,3] (shape: (3,)) is that extra outer bracket
- When I see [1] inside, that means the last shape number is 1 (a single element array)
- Arrays like (1, 2, 2) are just 2×2 matrices with an extra wrapper

My Journey Through Examples
I successfully analyzed 11 different array structures, starting with simple 3D arrays and working up to more complex patterns. Each one taught me something new:

1. The "chocolate bar" analogy helped me visualize 3D arrays as packs of chocolate bars
2. The "classroom with rows of desks" made shapes like (2, 3, 4) intuitive
3. Practicing with both formatted and messy arrays built my confidence
4. Discovering the .squeeze() method showed me how to remove unnecessary dimensions

Most Valuable Insights
1. Python will always tell me the shape → I don't need to strain my eyes reading messy arrays
2. Every dimension has meaning → First number = how many blocks, second = rows in each block, third = columns in each row
3. Real-world analogies are powerful → Bookshelves, classrooms, and chocolate bars make abstract concepts concrete
4. Practice builds pattern recognition → What was confusing yesterday is obvious today

The Confidence Boost
What started as a struggle ("horizontal vs vertical shapes confuse me") turned into a strength. I can now:
- Look at [[[1,2],[3,4]],[[5,6],[7,8]]] and immediately see (2, 2, 2)
- Understand why [[[1,2,3]]] has shape (1, 1, 3)
- Recognize when an array has unnecessary dimensions that can be squeezed


Mastering Array Slicing & Indexing

The Slicing Syntax That Finally Made Sense
After confusion with slice notation, I cracked the code: arr[start:stop:step]
- start → Where to begin (inclusive)
- stop → Where to end (exclusive - doesn't include this index!)
- step → How many to skip (default is 1)

Key slicing patterns I now use confidently:
- arr[2:5] → Get elements at index 2, 3, 4 (stops BEFORE 5)
- arr[:3] → First 3 elements (0, 1, 2)
- arr[3:] → From index 3 to the end
- arr[::2] → Every other element (step of 2)
- arr[::-1] → Reverse the array! (negative step)
- arr[-3:] → Last 3 elements

The "Stop is Exclusive" Revelation
This was my biggest "aha!" moment: arr[1:4] gives you 3 elements (indices 1, 2, 3), NOT 4!
Think of it as: "start at 1, go UP TO but not including 4"

Multidimensional Slicing - The Game Changer
For 2D arrays, I learned each dimension gets its own slice separated by commas:
- arr[0:2, 1:3] → Rows 0-1, Columns 1-2
- arr[:, 0] → ALL rows, first column only
- arr[1, :] → Second row, ALL columns
- arr[::2, ::2] → Every other row AND every other column

My mental model: arr[rows_slice, columns_slice, ...]

Advanced Slicing Tricks I Discovered
1. Negative indexing: arr[-1] gets the last element, arr[-2] gets second-to-last
2. Boolean indexing: arr[arr > 5] returns all elements greater than 5
3. Fancy indexing: arr[[0, 2, 4]] gets elements at specific indices
4. Combining slices: arr[1:4, [0, 2]] gets rows 1-3, but only columns 0 and 2

The Slicing vs Indexing Distinction
- Indexing: arr[2] → Returns a SINGLE element (reduces dimensions)
- Slicing: arr[2:3] → Returns an ARRAY with one element (preserves dimensions)
This matters for shape preservation!



Other Essential NumPy Concepts I Mastered

Array Creation Methods
- np.array([1,2,3]) → From Python list
- np.zeros((3,4)) → 3×4 array of zeros
- np.ones((2,3)) → 2×3 array of ones
- np.arange(0, 10, 2) → [0, 2, 4, 6, 8] (like range but returns array)
- np.linspace(0, 1, 5) → 5 evenly spaced numbers from 0 to 1
- np.eye(3) → 3×3 identity matrix

Reshaping Arrays
- arr.reshape(3, 4) → Change shape to 3 rows, 4 columns (must have same total elements!)
- arr.flatten() → Convert any shape to 1D
- arr.ravel() → Flattens but returns a view (more efficient)
- arr.T or arr.transpose() → Swap rows and columns

Array Operations (Broadcasting!)
I learned that NumPy automatically "broadcasts" operations across arrays:
- arr + 5 → Adds 5 to EVERY element
- arr * 2 → Multiplies EVERY element by 2
- arr1 + arr2 → Element-wise addition (if shapes are compatible)

The magic of broadcasting: NumPy stretches smaller arrays to match larger ones!

Key Array Attributes I Use Daily
- arr.shape → Get dimensions (e.g., (3, 4, 2))
- arr.ndim → Number of dimensions (e.g., 3)
- arr.size → Total number of elements
- arr.dtype → Data type (int64, float64, etc.)

Useful Array Methods
- arr.max(), arr.min() → Find maximum/minimum
- arr.sum() → Sum all elements
- arr.mean() → Average value
- arr.std() → Standard deviation
- arr.argmax(), arr.argmin() → INDEX of max/min element

Axis Operations - The Concept That Clicked
When I use axis parameter, I'm specifying WHICH dimension to operate along:
- arr.sum(axis=0) → Sum DOWN the rows (collapse rows)
- arr.sum(axis=1) → Sum ACROSS the columns (collapse columns)
Mental trick: The axis you specify is the one that DISAPPEARS!

Common Pitfalls I Learned to Avoid
1. Forgetting slices are VIEWS not copies → Modifying a slice changes the original!
   - Use arr.copy() to get an independent copy
2. Shape mismatches in operations → Always check arr.shape before combining arrays
3. Integer vs float division → np.array([1,2,3]) / 2 gives floats, not integers
4. Mutable default arguments → Never use arr=np.array([]) as a default parameter!



Personal Takeaways
- Don't fight messy formatting → Use np.array(arr).shape or reformat the code
- Think in hierarchies → Outer containers hold inner containers hold elements
- Every bracket has purpose → They're not just decoration
- Progress comes from repetition → Each example made me sharper
- Slicing is more powerful than looping → Master arr[start:stop:step] to avoid slow loops
- When stuck, print the shape → arr.shape reveals everything about the structure

Looking Ahead
Today built a solid foundation for tackling 4D arrays and beyond. The mental model is in place: if 3D arrays are like stacks of 2D matrices, then 4D arrays are like bookshelves of those stacks. I'm ready to explore color images (3D: height × width × RGB) and videos (4D: frames × height × width × RGB).

With slicing mastered, I can now efficiently extract regions of interest from images, select time ranges from datasets, and manipulate data without writing complex loops. NumPy's vectorized operations will make my code both faster and more readable.

Final Reflection
The breakthrough wasn't just about arrays—it was about learning how to learn complex concepts. By breaking things down, using analogies, and practicing relentlessly, I transformed confusion into competence. Understanding that slicing returns views (not copies) and that the stop index is exclusive were the final pieces that made everything click. This pattern will serve me well for every future programming challenge.

Next up: Broadcasting rules, advanced indexing techniques, and applying these skills to real data science problems!