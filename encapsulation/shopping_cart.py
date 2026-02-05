class ShoppingCart:
    def __init__(self, discount_percent=0.0):
        self._items = {}
        self._discount_percent = discount_percent 

    @property
    def items(self):
        return self._items.copy()
    
    @property
    def discount_percent(self):
        return self._discount_percent
    
    @discount_percent.setter
    def discount_percent(self, value):
        if value < 0:
            print(f'Discount cannot be negative')
            return
        if value > 100:
            print(f'Discount cannot be more that 100%')
            return
        else:
            self._discount_percent = value            

    def add_item(self, name, price):
        self._items[name] = price
        print(f'{name} has been added to the cart✅')

    def remove_item(self, name):
        if name in self._items: 
            del self._items[name]
            return
        else:
            print(f'Error')
    
    def get_subtotal(self):
        total = sum(self._items.values())
        print(f'The price of all items is {total}')
        return total
    
    def get_total(self):
        total = sum(self._items.values())
        discount_amount = total * (self._discount_percent) / 100
        final_total = total - discount_amount
        return final_total
    
    def __str__(self):
        return f'{self._items}, {self._discount_percent}% {sum(self._items.values())}'


# Test
if __name__ == "__main__":
    cart = ShoppingCart()
    cart.add_item('Laptop', 1000)
    cart.add_item('Mouse', 50)
    print(cart.get_subtotal())
    
    cart.discount_percent = 10
    print(cart.get_total())
    
    cart.discount_percent = 150
    print(cart)