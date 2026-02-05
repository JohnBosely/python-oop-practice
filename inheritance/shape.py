class Shape:
    def __init__(self, color):
        self.color = color
    
    def describe(self):
        return f'The color is {self.color}'


class Circle(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius
        
    def area(self):
        area_of_circle = 3.142 * self.radius ** 2
        return f'The area of the circle is {area_of_circle}'


class Rectangle(Shape):
    def __init__(self, color, width, height):
        super().__init__(color)
        self.width = width
        self.height = height

    def area(self):
        area_of_rectangle = self.width * self.height
        return f'The area of the rectangle is {area_of_rectangle}'


# Test
if __name__ == "__main__":
    circle = Circle('red', 5)
    print(circle.describe())
    print(circle.area())
    
    rectangle = Rectangle('blue', 4, 6)
    print(rectangle.describe())
    print(rectangle.area())