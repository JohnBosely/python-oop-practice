# class Vehicle:
#     def __init__(self, brand, model):
#         self.brand = brand
#         self.model = model
#         self.odometer = 0
    
#     def drive(self, distance):
#         self.odometer += distance
#         print(f'Drove {distance}km. Total: {self.odometer}km')


# class ElectricCar(Vehicle):
#     def __init__(self, brand, model, battery_size):
#         super().__init__(brand, model)
#         self.battery_size = battery_size
#         self.battery_level = 100

#     def drive(self, distance):
#         super().drive(distance)
#         self.battery_level -= distance / 10
#         print(f'Battery: {self.battery_level}% remaining')

# tesla = ElectricCar('Tesla', 'Model 3', 75)
# tesla.drive(50)
# # Expected output:
# # Drove 50km. Total: 50km
# # Battery: 95% remaining


class BankAccount:
    def __init__(self, account_holder, balance=0):
        self._account_holder = account_holder
        self._balance = balance
        self._transaction_count = 0
    
    @property
    def account_holder(self):
        return self._account_holder
    
    @property
    def balance(self):
        return self._balance
    
    def get_available_funds(self):
        """Default behavior: you can only spend what you have."""
        return self._balance
    #just applied this, i was told to do this, i didnt thinl of this

    def deposit(self, amount):
        if amount <= 0:
           print(f'Invalid Amount')
           return False
        else: 
            self._balance += amount
            self._transaction_count += 1
            print(f'You have deposited {amount} your balance is {self._balance}')
            return True
        
    def withdraw(self, amount):
        if amount <= 0:
            print(f'Invalid Amount')
            return
        
        if self._balance >= amount:
            self._balance -= amount
            self._transaction_count += 1
            print(f'You have withdrawn {amount}, you have {self._balance}')

class SavingsAccount(BankAccount):
    def __init__(self, account_holder, balance=0, interest_rate=0.02):
        super().__init__(account_holder, balance)
        self.interest_rate = interest_rate

    def add_interest(self):
        self._balance += (self._balance * self.interest_rate )
        print(f'Your balance is {self._balance} at an interest rate of {self.interest_rate}')     

    def withdraw(self, amount):
        if self.balance >= amount + 2:
            self._balance = self._balance - (amount + 2)
            print(f'{self._account_holder} has withdrawn {self._balance} from savings account')

class CheckingAccount(BankAccount):
    def __init__(self, account_holder, balance=0, overdraft_limit=100):
        super().__init__(account_holder, balance)
        self.overdraft_limit = overdraft_limit

    def withdraw(self, amount):
        funds = self._balance + self.overdraft_limit

        if amount > 0 and amount <= funds:
            self._balance -= amount
            print(f"Withdrew ${amount}. New balance: ${self._balance}.")
            return True
        elif amount > funds:
            print(f"Insufficient funds. Attempted to withdraw ${amount}, but only ${funds} available (including overdraft).")
            return False
        else:
            print("Invalid withdrawal amount.")
            return False
        
    def get_available_funds(self):
        return self._balance + self.overdraft_limit

savings = SavingsAccount('Alice', 1000, 0.05)
savings.add_interest()  # Add 5% = $50
savings.withdraw(50)    # Withdraws $52 (includes $2 fee)

checking = CheckingAccount('Bob', 100, 200)
print(checking.get_available_funds())  # 300
checking.withdraw(250)  # Works! Goes to -150
