class User:
    def __init__(self, username, password):
        self._username = username
        self._password = password
    
    @property
    def username(self):
        return self._username
    
    def set_password(self, new_password):
        new_password = str(new_password)
        if len(new_password) < 8:
            print('Password is less than 8 characters')
            return
        else:
            print('Password is strong')
            self._password = new_password

    def check_password(self, password):
        if password == self._password:
            return 'Password is correct'
        else:
            return 'Password is incorrect'
        
    def __str__(self):
        return f'Username: {self._username}, Password:********'


# Test
if __name__ == "__main__":
    user = User('John_Doe', 'secret123')
    print(user)
    print(user.check_password('wrong'))
    print(user.check_password('secret123'))
    
    user.set_password('new')
    user.set_password('newpassword')
    print(user.check_password('secret123'))
    print(user.check_password('newpassword'))