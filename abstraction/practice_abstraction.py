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
        


#MORE ABSTRACTION EXCERCISES
from abc import ABC, abstractmethod

class Database(ABC):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._connected = False

    @abstractmethod
    def connect(self):
        #establish connection
        pass

    @abstractmethod
    def disconnect(self):
        #close connection
        pass

    @abstractmethod
    def execute_query(self, sql):
        self.sql = sql
        #run sql query
        pass

    def get_status(self): #why is a connected attribute not here, when you said based on the _connected attribute
        if self._connected: #this part makes no sense cos if self._connected is false, why is it true here
            return 'Connected'
        else:
            return 'Disconected'
        
    

class MySQL(Database):
    def __init__(self, host, port=3306):
        super().__init__(host, port)

    def connect(self):
        self._connected = True
        print (f'Connecting to MySQL at localhost: {self.host}...')#tbh i dont know this self port stuff so i dont knw what to type to show the user

    def disconnect(self):
        self._connected = False
        print (f'Disconnecting from MySQL at local host: {self.host}')

    def execute_query(self, sql):
        self.sql = sql
        return (f'Executing MySQL query {sql}')
   


class PostgreSQL(Database):
    def __init__(self, host, port=5432):
        super().__init__(host, port)

    def connect(self):
        self._connected = True
        print (f'Connecting to PostgreSQL at localhost: {self.host}...')#tbh i dont know this self port stuff so i dont knw what to type to show the user

    def disconnect(self):
        self._connected = False
        print (f'Disconnecting from PostgreSQL at local host: {self.host}')

    def execute_query(self, sql):
        return (f'Executing PostgreSQL query {sql}')
    
    

class MongoDB(Database):
    def __init__(self, host, port=27017):
        super().__init__(host, port)

    def connect(self):
        self._connected = True
        print (f'Connecting to MongoDB at localhost: {self.host}...')#tbh i dont know this self port stuff so i dont knw what to type to show the user

    def disconnect(self):
        self._connected = False
        print (f'Disconnecting from MongoDB at local host: {self.host}')

    def execute_query(self, sql):
        return (f'Executing MongoDB query {sql}')
    
    

from abc import ABC, abstractmethod

class Notification(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def send(self, recipient, message):
        #Send Notification
        pass

    @abstractmethod
    def format_message(self, message):
        #Format Message
        pass

    def log_notification(self, recipient):
        print(f'Notification sent to {recipient}')

class EmailNotification(Notification):
    def __init__(self):
        super().__init__()
    
    def format_message(self, message):
        return (f'Subject: Notification | "{message}"') 

    def send(self, recipient, message):
        formatted = self.format_message(message)
        print(f'Formatted: {formatted}')
        print(f'Sending email to {recipient}')
        self.log_notification(recipient)  

class SMSNotification(Notification):
    def __init__(self):
        super().__init__()

    def format_message(self, message):
        if len(message) > 160:
            return message[:160]
        else:
            return message
        # or def format_message(self, message):
        #       return message[:160]

    def send(self, recipient, message):
        formatted = self.format_message(message)
        print(f'Formatted: {formatted}')
        print (f'Sending SMS to {recipient}')
        self.log_notification(recipient) 
    
class PushNotification(Notification):
    def __init__(self, app_name):
        super().__init__()
        self.app_name = app_name
    
    def format_message(self, message):
        return (f'[{self.app_name}]: {message}')
    
    def send(self, recipient, message):
        formatted = self.format_message(message)
        print(f'Formatted: {formatted}')
        print(f'Sending push notification to {recipient}')
        self.log_notification(recipient) 


    
