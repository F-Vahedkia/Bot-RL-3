# Run: python -m temp2

from dataclasses import dataclass

@dataclass(frozen=True)
class Symbol:
    name: str

    def __post_init__(self):
        name = self.name.upper().strip()

        object.__setattr__(
            self,
            "name",
            name
        )
s = Symbol(" eurusd ")
print(s.name)




class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height


@property
def age(self):
    return self._age

@age.setter
def age(self, value):
    if value < 0:
        raise ValueError("age cannot be negative")
    self._age = value











class A:
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value


