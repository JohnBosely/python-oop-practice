class StudentManagementSystem:
    def __init__(self):
        self._students = {}

    @property
    def students(self):
        return self._students.copy()
    
    def add_student(self, student_id, name, age):
        self._students[student_id] = {
            'name': name,
            'age': age,
            'courses': []
        }
        print(f'Added student: {name}')
        return self._students
    
    def remove_student(self, student_id):
        if student_id in self._students:
            name = self._students[student_id]['name']
            del self._students[student_id]
            print(f'Removed student: {name}')
        else:
            print(f'Student {student_id} not found')
        
    def get_student(self, student_id):
        return self._students.get(student_id)

    def enroll_course(self, student_id, course_name):
        if student_id not in self._students:
            print(f'Student ID {student_id} not found')
            return
        if 'courses' not in self._students[student_id]:
            self._students[student_id]['courses'] = []

        if course_name not in self._students[student_id]['courses']:
            self._students[student_id]['courses'].append(course_name)
            print(f"Successfully enrolled in {course_name}")
        else:
            print(f'Student is already enrolled in this course')
        
    def drop_course(self, student_id, course_name):
        if course_name in self._students[student_id]['courses']:
            self._students[student_id]['courses'].remove(course_name)
            print(f'You have dropped {course_name}')
        else:
            print(f'You have not selected this course')

    def get_all_students(self):
        names = []
        for student in self._students.values():
            names.append(student['name'])
        return names
    
    def get_students_in_course(self, course_name):
        students_in_course = []
        for student_id, student_info in self._students.items():
            if course_name in student_info['courses']:
                students_in_course.append(student_info['name'])
        return students_in_course
            
    def __str__(self):
        return f'{self._students}'


# Test
if __name__ == "__main__":
    system = StudentManagementSystem()
    system.add_student('S001', 'Alice', 20)
    system.add_student('S002', 'Bob', 20)
    system.add_student('S003', 'Charlie', 20)
    
    system.enroll_course('S001', 'Math')
    system.enroll_course('S001', 'Science')
    system.enroll_course('S002', 'Math')
    system.enroll_course('S003', 'English')
    
    print(system.get_student('S001'))
    
    system.remove_student('S002')
    print(system)