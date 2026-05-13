#first k prime numbers for input validaiton

while True:
    k = int(input("Please input k: "))
    if k > 0:
        break
    print("Please enter a positive num")

primes = []

n = 2

while len(primes) < k: # we need k numbers
    is_prime = True
    for d in range(2, int( n ** 0.5) + 1):
        if n % d == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(n)
    n+=1
print("Prime numbers: ", *primes)

# to soolve the prime problem
