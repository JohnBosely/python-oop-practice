class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def info(self):
        return f'This is a {self.brand} {self.model}'


class Car(Vehicle):
    def __init__(self, brand, model, num_doors):
        super().__init__(brand, model)
        self.num_doors = num_doors


class Motorcycle(Vehicle):
    def __init__(self, brand, model, has_sidecar):
        super().__init__(brand, model)
        self.has_sidecar = has_sidecar


# Test
if __name__ == "__main__":
    car = Car('Toyota', 'Camry', 4)
    print(car.info())
    print(f'Doors: {car.num_doors}')
    
    bike = Motorcycle('Harley', 'Sportster', False)
    print(bike.info())
    print(f'Has sidecar: {bike.has_sidecar}')