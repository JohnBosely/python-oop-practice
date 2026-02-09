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
Encapsulation** - Private attributes, properties, getters/setters
Data Structures - Lists, dictionaries, sets, tuples
Inheritance - Parent/child classes, `super()`

HOW TO RUN
Ech file can basically be run independently, you dont need any fancy thing to run it, just use bash.

python encapsulation/bank_account.py
python data_structures/inventory_tracker.py
python inheritance/vehicle.py

KEY CONCEPTS I LEARNED
OOP was quite easy for me to understand foe some weird reason, genuinely dont know how i understood it so well. I'll probably say its because of how fun i made it, i did a few projects (not here though) using game scenarios, it involved the characters health, stats, and other stuff. Gamification of topics really helped me, brocode and Mosh were so so important also.

This is a summary of what i learnt
Encapsulation
 Using `_` prefix for private attributes
 Creating `@property` decorators
 Implementing setters with validation
 Data protection

 Data Structures
 Nested dictionaries
 Dictionary methods (`.items()`, `.values()`, `.keys()`)
 List comprehensions
 CRUD operations

 Inheritance
 Parent/child relationships
 Using `super()`
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
