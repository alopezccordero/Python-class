# object oriented programming part 1
## object:
an object is an entity with a state, behavious, often defined with a class.

Each ooobject is an instance of its class.

* atributes:
 - __init__(self).
 the self parameter refers to the same object.
 - instance attribute. defined inside the __init__
 - class attribute. shared by all instances.
## class functions for attributes:
 * hasattr(x, "attr") looks for an attribute
 * getattr(x, "attr") looks for attr value
 * setattr(x, "attr", value): sets the value to the attribute of that instance
 * delattr(x, "attr"), deletes that attr.

## inheritance
 this is when a class inherits the properties of another class
  - thechnicality: the subclass inherits the super class or parent class.

### method ovveriding when inheriting
 - super is used with the next syntax:
  def __init__(self, attr_from_parent):
    super().__init__(attr_from_parent)
    this code makes the child class inherit the fields from upper class
 * a class also has an __iter__ method
 which is an object that produces elements via next() function.
## key takeaways
 * __init__ runs automatically when object is created, however, __init__ doesnt have to be defined in every class
 * self is the object itself. this is a norma. objects always take self.
 * class attr. this is shared by all instances. 
 * instance attr. this is unique to each instance.
 * use super().method() to call the parents version of an overriding method.
 * an iterable can be used in a for loop. an interator produces one val at a time with method next().

## modules and packages.
 - we import modules with import, from, as.
 a module is a .py file that can e imported.
 - __name__ = "__main__" runs only when executed directly on the file that was defined. 
 we usually run:
 main()

### numpy module.
    this is foundational for scientific computing and provides a wide range of mathematical functions written in c.
 - norm is to import numpy as np.
 - we can use vectorized operations with numpy.
 - check numpy_labs.py for more information

## key takeaways
* a module is a .py file. a package is a directory of modules
* three import forms. import x, import x as y, from x import y.
* if __name == "__main__"; runs code only when the file is executed directly.
* numpy's main advantage is vectorization.

# numpy - deeper understanding
 lets say we have two arrays
 a = np.array([1, 2, 3])
 b = np.array([[1, 2], [3, 4]])

 we can use the next functions
 * shape:
  - a.shape will output 3
  - b.shape will output 2, 2. 
  shape function outputs the dimension of an array
 * size:
  - a.size will output number of elements in a.
  - b.size will output number of elements in b.
  size outputs number of elements.
## array creation with numpy.
* np.zeros((rows, cols)) gives an awway os zeros. then we have np.ones
* we also have np.random.rand in [0, 1)
* np.arrange(start, stop, step). like pythons range buy returns an array.
* np.linspace(start, stop, num) evenly spaced values from start to stop.
## reshape with numpy
* np.arrange to arrange vals
* var.reshape(a, b) makes a rows and b cols

##operations
* operations are performed with linear algebra.
* dot product. this is a multiplication operation that returns one single value.

## slicing with np

examples: 
 - print(M[0, 1]) prints row 0 col 1
 - print(M[:, 0]) prints all values from column 0.
 - print(M[1, :]) print row 1( second row)
 - print(M[0:2, 1:3]) submatrix with row 0, 1 and columns 1 and 3:
 outputs: 2, 3. 5, 6.
## other functions
 * transpose is print(var.T)
 * np.sum(a)
 * np.mean(a)
 * np.std(a) standard deviation
 * np.concatenate([a, b]) join arrays
 * np.sort(a)
For more information review numpy_basic_operations.py

## key takeaways
* shape gives dimensions, size gives total of elements
* element wise vs matrix operations: * is element wise. np.dot is the dot/matrix prouct
* reshape keeps the same total of elements.
* slicing 2d arrays uses [rows, cols]