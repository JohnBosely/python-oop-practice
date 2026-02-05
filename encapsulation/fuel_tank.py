class FuelTank:
    def __init__(self, capacity, current_fuel=0.0):
        self._capacity = max(1.0, float(capacity))
        self._current_fuel = max(0.0, min(float(current_fuel), self._capacity))

    @property
    def capacity(self):
        return self._capacity

    @property
    def current_fuel(self):
        return self._current_fuel

    @current_fuel.setter
    def current_fuel(self, value):
        value = float(value)
        if value < 0:
            raise ValueError("Fuel cannot be negative!")
        if value > self._capacity:
            raise ValueError(f"Fuel cannot exceed capacity ({self._capacity}L)")
        self._current_fuel = value
        print(f"Fuel updated to {self._current_fuel:.1f}L")

    def fill(self, amount):
        if amount <= 0:
            print("Can't add zero or negative fuel!")
            return
        new_fuel = self._current_fuel + amount
        if new_fuel > self._capacity:
            extra = new_fuel - self._capacity
            self._current_fuel = self._capacity
            print(f"Tank full! Added {amount:.1f}L (wasted {extra:.1f}L)")
        else:
            self._current_fuel = new_fuel
            print(f"Added {amount:.1f}L. Now {self._current_fuel:.1f}L")

    def drive(self, distance):
        if distance <= 0:
            print("Can't drive zero or negative distance!")
            return
        fuel_needed = distance / 10.0
        if self._current_fuel >= fuel_needed:
            self._current_fuel -= fuel_needed
            print(f"Drove {distance}km. Used {fuel_needed:.1f}L.")
        else:
            max_km = self._current_fuel * 10
            self._current_fuel = 0
            print(f"Not enough fuel! Drove {max_km:.1f}km and ran out.")

    @property
    def percent_full(self):
        return (self._current_fuel / self._capacity) * 100

    def __str__(self):
        return f"Fuel Tank: {self._current_fuel:.1f}/{self.capacity:.1f}L ({self.percent_full:.1f}%)"


# Test
if __name__ == "__main__":
    tank = FuelTank(50, 15)
    print(tank)
    
    print(tank.current_fuel)
    tank.current_fuel = 40
    print(tank)
    
    tank.fill(30)
    print(tank.percent_full)