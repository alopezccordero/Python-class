## start with left aligned increasing triangle

n = int(input("pleaser enter an input: "))

for i in range(1, n+1):
    for j in range(i):
        print("*", end="")

    print()

#right aligned triangle

for i in range(1, n+1):
    for j in range(n - i):
        print(" ", end="")
    for j in range(i):
        print("*", end="")
    print()

#right aligned decreasing triangle

print()
for i in range( n+1, 1, -1):
    for j in range(i, 1, -1):
        print("*", end="")
    print()