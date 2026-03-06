import math as m
def getSteps(firstP, secondP):
    x1, y1 = firstP[0], firstP[1]
    x2, y2 = secondP[0], secondP[1]
    #try is used to avoid the division by 0.

    
    distance = m.floor((abs(x2 - x1) + abs(y2 - y1))/2)
    return distance
    
    


file = open("input.txt", 'r')

lines = file.readlines()

n, k = map(int, lines[0].split())
##n = square size, k = number of civilizations
civilizations = []
civ = [None, None]

for x in range(1, 1+k):
    a, b= map(int, lines[x].split())#parse to int
    
    civilizations.append([a, b])#matrix with civilizations

optimalConnections = {}

print(civilizations)
for civil in civilizations:
    MinimunSteps = 100
    for x in civilizations:  
        if x == civil:
            continue          
        print(f"steps between {civil} and {x} is: ")
        print(f"{getSteps(x, civil)}")
        CurrentSteps = getSteps(civil, x)
        if CurrentSteps < MinimunSteps:
            MinimunSteps = CurrentSteps
    optimalConnections.update({f"{civil}": MinimunSteps})
        
print(optimalConnections)

maxSteps = -1
for key in optimalConnections:
    if maxSteps < optimalConnections.get(key):
        maxSteps = optimalConnections.get(key)
print(maxSteps)
output = open("output.txt", 'w')
output.write(f"{maxSteps}")
output.close()
file.close()