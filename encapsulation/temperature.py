class Temperature:
    def __init__(self, celsius=0.0):
        self._celsius = celsius

    @property
    def celsius(self):
        print(f'Calculating conversion... Done!')
        return self._celsius
        
    @property
    def fahrenheit(self):
        print(f'Calculating conversion... Done!')
        return (self._celsius * 9/5) + 32
    
    @property
    def kelvin(self):
        print(f'Calculating conversion... Done!')
        return self._celsius + 273.15

    @celsius.setter
    def celsius(self, value):
        value = float(value)
        if value < -273.15:
            print(f'It is lower than absolute zero')
            return
        else:
            self._celsius = value 
    
    def __str__(self):
        return f''


# Test
if __name__ == "__main__":
    temp = Temperature(25)
    print(temp.celsius)
    print(temp.fahrenheit)
    print(temp.kelvin)
    temp.celsius = 100
    print(temp.fahrenheit)
    temp.celsius = -500