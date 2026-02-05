class Student:
    def __init__(self, name):
        self._name = name
        self._grades = []

    @property
    def name(self):
        return self._name
    
    @property
    def grades(self):
        return self._grades.copy()
    
    def add_grade(self, grade):
        if grade < 0:
            print(f'Invalid Score')
            return
        if grade > 100:
            print('The highest is 100, invalid score')
            return
        if grade > 0 and grade <= 100:
            self._grades.append(grade)

    def get_average(self):
        if not self._grades:
            return 0.0
        average = sum(self._grades) / len(self._grades)
        print(f'The average: ')
        return average

    def get_letter_grade(self):
        current_average = self.get_average()
        if current_average < 0:
            return 'Grade cannot be negative'
        if current_average >= 90:
            return 'A'
        elif current_average >= 80:
            return 'B'
        elif current_average >= 70:
            return 'C'
        elif current_average >= 60:
            return 'D'
        else:
            return 'F'
        
    def __str__(self):
        return f'{self._name}, your score is {self._grades}'


# Test
if __name__ == "__main__":
    student = Student('Bob')
    student.add_grade(85)
    student.add_grade(92)
    student.add_grade(78)
    student.add_grade(150)
    print(student.get_average())
    print(student.get_letter_grade())
    print(student)