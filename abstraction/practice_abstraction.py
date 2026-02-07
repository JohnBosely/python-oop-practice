# from abc import ABC, abstractmethod

# class Shape(ABC):
#     @abstractmethod
#     def area(self):
#         pass

#     @abstractmethod
#     def perimeter(self):
#         pass

#Abstraction is the hiding of complexity, it only shows the users what is needed.
#There are some methods you need and if you dont use abstraction you could foget them and have to write them every single time(if you use inheritance)
#e.g. a remote hides the complexity of turning on or off a TV, but every remote must know how to do this functions, so you make the on or off methods abstract ones so you can use them anywhere, its that simple to be honest.

#PRACTICE QUESTIONS
from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    @abstractmethod
    def start_engine(self):
        pass

    @abstractmethod
    def stop_engine(self):
        pass

    @abstractmethod
    def get_fuel_type(self):
        pass

    def honk(self):
        return (f'Beep Beep')

class ElectricCar(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand, model)

    def start_engine(self):
        return (f'Electric motor started silent')


    def stop_engine(self):
        return (f'Electric motor stopped silent')

    def get_fuel_type(self):
        return (f'Electric')
        

class GasCar(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand, model)

    def start_engine(self):
        return (f'Vroom! Engine started')

    def stop_engine(self):
        return (f'Engine stopped')


    def get_fuel_type(self):
        return (f'Gasoline')
        

class HybridCar(Vehicle):
    def __init__(self, brand, model):
       super().__init__(brand, model)

    def start_engine(self):
        pass #no need to write anything, just practicing

    def stop_engine(self):
        pass

    def get_fuel_type(self):
        return (f'Hybrid')


# vehicle = Vehicle('Toyota', 'Generic')  # Should ERROR

electric = ElectricCar('Tesla', 'Model 3')
print(electric.start_engine())  # "Electric motor started silently"
print(electric.get_fuel_type())  # "Electric"
electric.honk()  # "Beep beep!"

gas = GasCar('Ford', 'Mustang')
print(gas.start_engine())  # "Vroom! Engine started"
print(gas.get_fuel_type())  # "Gasoline"
        
