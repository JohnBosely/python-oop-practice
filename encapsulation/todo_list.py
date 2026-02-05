class TodoList:
    def __init__(self, title):
        self._tasks = []
        self._title = title

    @property
    def title(self):
        return self._title
    
    @property
    def tasks(self):
        return self._tasks[:]

    def add(self, task):
        self._tasks.append(task)
        print(f'Added: {task}')

    def remove(self, task):
        if task in self._tasks:
            self._tasks.remove(task)
            print(f'Removed: {task}')
        else:
            print(f'Task: {task}\nis not in the tasks')

    def mark_done(self, task): 
        if task in self._tasks:
            self._tasks.remove(task)
            print(f'Marked as done: {task} ✅')
                   
    def clear_all(self):
        self._tasks = []
        print(f'All tasks cleared!')
    
    def __str__(self):
        return f'Todo List: {self.tasks}'


# Test
if __name__ == "__main__":
    my_todo = TodoList("Daily Tasks")
    print(my_todo)
    
    my_todo.add("Buy groceries")
    my_todo.add("Call mom")
    my_todo.add("Finish OOP practice")
    print(my_todo)
    
    my_todo.mark_done("Call mom")
    print(my_todo)
    
    my_todo.remove("Buy groceries")
    print(my_todo)
    
    my_todo.remove("Go to gym")
    print(my_todo)
    
    my_todo.add("Read book")
    print(my_todo)
    
    my_todo.clear_all()
    print(my_todo)