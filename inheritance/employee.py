class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def get_info(self):
        return f'My name is {self.name}, I earn ${self.salary}'


class Manager(Employee):
    def __init__(self, name, salary, team_size):
        super().__init__(name, salary)   
        self.team_size = team_size

    def give_raise(self, amount):
        self.salary += amount
        print(f'{self.name} got a ${amount} raise! New salary is ${self.salary}')


class Developer(Employee):
    def __init__(self, name, salary, programming_language):
        super().__init__(name, salary) 
        self.programming_language = programming_language
    
    def __str__(self):
        return f'My name is {self.name}, I am a {self.programming_language} programmer that earns ${self.salary}'


# Test
if __name__ == "__main__":
    dev = Developer('David', 300000, 'Python')
    print(dev)
    
    manager = Manager('Alice', 80000, 5)
    print(manager.get_info())
    manager.give_raise(5000)