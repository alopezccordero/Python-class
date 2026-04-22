class TechWorker:
    def write_code(self):
        print(f" is writing code")
    def lead_team(self):
        print(f" is leading the team")
    def creating_budget(self):
        print(f" is creating the report")
class Scrum_Master(TechWorker):
    def initialize(self):

        print("the scrum master is: ")
    def write_code(self):
        raise Exception("Scrum master should not write code")
        #Scrum master should not write code
    def lead_team(self):
        return super().lead_team()
    def creating_budget(self):
        return super().creating_budget()

class SoftwareEngineer(TechWorker):
    def initialize(self):

        print("The software engineer is: ")
    def write_code(self):
        return super().write_code()
    def lead_team(self):
        raise Exception("SWE should not lead team")
        #the software engineer should not lead the team
    def creating_budget(self):
        raise Exception("SWE should not create budget")
        #The software engineer should not manage budget"
softwarengineer = SoftwareEngineer()
softwarengineer.initialize()
softwarengineer.write_code()

scrum_master = Scrum_Master()
scrum_master.initialize()
scrum_master.lead_team()
scrum_master.creating_budget()