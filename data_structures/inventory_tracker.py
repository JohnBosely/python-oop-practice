class InventoryTracker:
    def __init__(self):
        self._inventory = {}

    @property
    def inventory(self):
        return self._inventory.copy()
    
    def add_product(self, name, price, quantity):
        self._inventory[name] = {
            'price': price,
            'quantity': quantity
        }
        print(f'Added {name} to inventory')
        return name
    
    def remove_product(self, name):
        if name in self._inventory:
            del self._inventory[name]
            print(f'Removed {name} from inventory')
            return
        else:
            print(f'{name} is not in the cart')
            return

    def update_quantity(self, name, quantity):
        new_quantity = int(quantity)
        if quantity < 0:
            print(f'Quantity cannot be negative')
            return quantity
        if name in self._inventory:
            self._inventory[name]['quantity'] = new_quantity
            print(f'Updated {name} quantity to {new_quantity}')
            return quantity

    def restock(self, name, amount):
        if name in self._inventory:
            self._inventory[name]['quantity'] += amount
            restock_total = self._inventory[name]['quantity']
            print(f'Restocked {name}: {amount} units added (now {restock_total} total)')
            return 
        
    def sell(self, name, amount):
        present_product = self._inventory[name]['quantity']
        if present_product < amount:
            print(f'Error: Not enough {name} in stock, you have {present_product} left')
            return
        if name in self._inventory:
            present_product = self._inventory[name]['quantity']
            present_product -= amount
            self._inventory[name]['quantity'] = present_product
            print(f'Sold {amount} {name}(s). Remaining: {present_product}')
            return

    def get_product(self, name):
        if name in self._inventory:
            return self._inventory[name]
        
    def get_low_stock(self, threshold):
        names = []
        if threshold < 0:
            print(f'Threshold cannot be negative')
            return
        
        for product_name, product_info in self._inventory.items():
            if product_info['quantity'] < threshold:
                names.append(product_name)
        return names

    def get_total_value(self):
        total = 0
        for product_info in self._inventory.values():
            current_price = product_info['price']
            current_quantity = product_info['quantity']
            total_value = current_price * current_quantity
            total += total_value
        return total
    
    def __str__(self):
        result = 'Inventory:\n'
        for product_name, product_info in self._inventory.items():
            price = product_info['price']
            quantity = product_info['quantity']
            value = price * quantity
            result += f'{product_name} - Price: ${price}, Quantity: {quantity}, Value: ${value}\n'
        return result


# Test
if __name__ == "__main__":
    inventory = InventoryTracker()
    
    inventory.add_product('Laptop', 999.99, 10)
    inventory.add_product('Mouse', 29.99, 50)
    inventory.add_product('Keyboard', 79.99, 25)
    print(inventory)
    
    inventory.restock('Mouse', 20)
    
    inventory.sell('Laptop', 3)
    inventory.sell('Mouse', 80)
    
    inventory.update_quantity('Keyboard', 30)
    
    print(inventory.get_low_stock(15))
    
    print(f'Total value: ${inventory.get_total_value()}')
    
    print(inventory.get_product('Mouse'))
    
    print(inventory)
    
    inventory.remove_product('Keyboard')
    print(inventory)