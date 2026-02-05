class BankAccount:
    def __init__(self, account_holder, balance=0):
        self._account_holder = account_holder
        self._balance = balance
    
    @property
    def account_holder(self):
        return self._account_holder
    
    @property
    def balance(self):
        return self._balance
    
    def deposit(self, amount):
        if amount <= 0:
            print(f'Invalid Amount')
            return
        self._balance += amount
        new_balance = self._balance
        print(f'You have deposited ${amount}, your new balance is ${new_balance}')
    
    def withdraw(self, amount):
        if self._balance < amount:
            print(f'You cant withdraw, your balance is too low')
            return
        if self._balance >= amount:
            self._balance -= amount
        print(f'${amount} has been withdrawn from your account, Your balance is ${self._balance}')

    def __str__(self):
        return f'Account holder: {self._account_holder}, Balance: ${self._balance}'


# Test
if __name__ == "__main__":
    account = BankAccount('Alice')
    account.deposit(100)
    account.withdraw(30)
    account.withdraw(100)
    account.deposit(100)
    account.deposit(10000000)
    print(account)