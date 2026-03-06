#file manipulation is used with open()
#"r", this is used to read / if no file is found, the program returns error
#"a", this is used to append / if no file is found, the file is created
#"w", this is used to write / if no file is found- the file is created
#"x", this is used to create.

#f = open("test.txt", "a") ##a file is opened/created to append lines

#f.write("Hello, Alejandro Lopez!")

#f.close()#it is a good practice to close files

#f = open("test.txt", "r")
#print(f.read())

f = open("test.txt", "r")

lines = f.readlines()


print(lines[10])
f.close()
