# For loops and list comprehensions

## for loops:
 This is used to iteratee over a sequence
 * break: stops the loop entirely
 * continue: continues to the next iteration.
 * Nested loops: usually with I and J as iterators
 * end parameter in print
    When using print("something", end="") prints with no newline.
## key takeaways
 * in range(end) is exlusive of end
 * in range(start, end, step) for control
 * break exits loop, continue is for next iteration.
 * the inner loop runs fully for every outer loop.
 * expr for item in iterable. (understand this.)
 * understand asterisk problems

# while loops
 this means: as long as the condition is true.

## key takeaways
 * always remember to create a condition that can become false.
 * while true + break is the standard pattern for input validation and "play again" loops
 * prime check optimization only test divisors up to sqrt(n)

    - to solve the primes problem. for each number that we try to verify, we need to iterate 2 to that number to the -0.5 (meaning the square root of that number ) + 1. and verify if the remainder of n / d is 0. 
    - key formula: int(n ** 0.5) + 1.
    - remember: only test divisors up to sqrt(n) which is the same as ** 0.5
# function and scopes.
## use *args, **kwargs
 - args is that we can input as many parameters as we want.
 - kwargs. we can input as many parameters as we want. but each of them has a key and a value.

 - default parameter value is defined like this:
    def great(x="default")
    if we call the function. the val is default
## object references

### for variables
 - if we pass a variable as a parameter inside a function. it is locally defined, unless we return it.
 - if we pass a list as a parameter in a function, the global list will be modified.

## scope
 * variables inside a function are local
 * variables at the top are global.
 * global keyword lets you modify a vvariabble.
## lambda functions
 * we define a function as a variable with syntax: 
 x = lambda a: a + 10.
 print(x(5)). a is the parameter

 y = lambda a, b: a * b
 print((y(5, 6))). 
 * lambda is used to create a function in one line.

## key takeaways
 * parameters are definitions.
 * arguments are the values passed.
 * defining a function doesnt execute it
 * *args tuple of positional arguments.
 * **kwargs = dictionary of keywords.
 * immutable objects act by value
 * mutable obbjects can be modified in place
 * global when rebinding a global name. list and dictionaries do not require global.
 * lamnbda one line anonymous function with a single expression.