class HospitalEmployee:
    def treat_patient(self):
        pass
    def schedule_appointment(self):
        pass
    def process_billing(self):
        pass

class Doctor(HospitalEmployee):
    def treat_patient(self):
        print("Doctor Kim is diagnosing the pt")
    def schedule_appointment(self):
        print("Doctor Kim is scheduling an appointment for the pt")
    def billing(self):
        print("Doctor Kim is billing the pt")

class Receptionist(HospitalEmployee):
    def treat_patient(self):
        raise Exception("Receptionists should not treat patients")
    
    def schedule_appointment(self):
        print("Receptionist is scheduling the patient")
        
    def process_billing(self):
        print("Receptionist is billing the patient")