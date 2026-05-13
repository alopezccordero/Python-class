# review for final test

Python uses:
1. Machine Learning. Powerful frameworks for ml
2. Data Science. libraries like pandas
3. Web development. - large library ecosystem

## key takeaways: 
1. Guido van Rossum, the netherlands
2. Bytecode + virtual machine - interpeted with a twist

## variables
* what is it?:
it is a reserved memory location with a label. no declaration needed

* Data types:
 - int
 - float
 - string
 - list
 - tuple : ordered unchangeable collection
 - dictionary : a collection of key value pairs.

* Formatting
 - :.2 is to two decimal points
 - slicing: use [:] index starts from 0 from the front and -1 from the back.

* Operators:
 - +
 - '-'
 - *
 - /
 - %
 - **
 - // this is floor division
 - whe can put a = behind each operator to take into account the variable value

 - and, or, not, xor
 - in, not in.
 - is, not is. (if both vars refer to the same object)

 ### variables
 - .split() to read several vars in an input

 ### type casting
 - int()
 - float()
 - str()

 ## key takeaways:
 - name of all 7 operators and categories.
    * division
    * multiplication
    * addition
    * substraction
    * mod
    * floor division
    * exponentiation.
 - Difference between float division and floor division
    * floor division returns lower val
    * floor division returns decimal
- Difference between == and is:
    * == is: is equal to.
    * is. are we referring to the same objecT?
    * input always returns a string. we need to use casting
    * input().split() to obtain several vals at once.

- Formulas for cylinder volume and distance between two points
    * pi * radius^2 * h
    * distance between two points.
    * sqrt((x2 - x1)^2 + (y2 + y1)^2)

# tuples and dictionaries.

### list 
- Ordered and changeagble collection of data.
- we access it with indexes.
- list slicing:
    * thislist[2:5] starts from index 2 and does not include last index
    * thislist[3:] starts from index 3.
    * thislist[:4] from start - does not include index 4. finishes in index 3.
- negative indexing:
    *  -1 refers to last item, -2 to the second last.
    * thislist[-4:-1]. starts from index tot-4 to -1 (exclusive )
- functions include len, max, min, sorted and sum
- common methods:
    1. append, extend, insert, remove, pop, clear, sort.
### tuples
 - these are immutable. use parenthesis.
 - can contain different types
### dictionaries.
 - stores key value pair
 - values can be any type
 - use {}

## key takeaways:
 * list indexing is zero-based, negative indexing starts from -1 at the end.
 * slicing's end index is exclusive - start index is inclusive
 * list = ordered and mutable
 * tuple = ordered and inmutable
 * dictionarie = key - value pairs.
 * append addds one element, extend adds all elements of an iterable.

# conditional statements.
 * if, else , elif.
 * nested statements- statements inside one another
 * map is used with split
    - a, b, c = map(int, input().split()) - this is an example.

## random module:
 - import random.
 - we need to set a random seed.
 - random.randint(a, b) gives a random int between a and b

## Key takeaways
 * identention matters.
 * elif is checked once previous if is not true
 * random.randint(a, b) is inclusive.
 * map(int, input().split()) is common for multipe intergers
 - Formulas for this chapter:
    * triangle inequality formula:
        The sum of any two sides must be greater than the third
    * Shoelace formula:
        Area = 1/2 |x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)| - there are 3 points. the formula is basically the summation of the multiplication x point with the substraction of the remaining y points (exlusive). divided by 2. 