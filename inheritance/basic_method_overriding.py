class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f'{self.name} makes a sound'
    
class Dog(Animal):
    def speak(self):
        return f'{self.name} says woof'
    

class Cat(Animal):
    def speak(self):
        return f'{self.name} says Meow'
    
class Cow(Animal):
    def speak(self):
        return f'{self.name} says Moo'
    
dog = Dog('Buddy')
cat = Cat('Whiskers')
cow = Cow('Boo')
print(dog.speak())
print(cat.speak())
print(cow.speak())
    
    
        