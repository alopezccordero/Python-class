def FindInfected(infected, lines, patients):
    NewLines = lines

    UpdatedList = patients
    CurrentInfected = None
    for x in range(2, len(lines) - 1):
   
        if f"{infected}" in lines[x]:

            a, b = map(int, lines[x].split())
                   
            if a == infected:
                CurrentInfected = b
            elif b == infected:
                CurrentInfected = a
            
            if CurrentInfected not in patients and CurrentInfected is not None:
                patients.append(CurrentInfected)
                
                NewLines[x] = "USED"
                g = FindInfected(CurrentInfected, NewLines, patients)
            CurrentInfected = None
        
    return UpdatedList, NewLines
        
f = open("input.txt", "r")

lines = f.readlines()
Patient_Z = int((lines[-1]))
patients = [(int(lines[-1]))]

patients, lines = FindInfected(Patient_Z, lines, patients)

output = open("output.txt", "w")
output.write(f"{len(patients) - 1}")

f.close()
output.close()
