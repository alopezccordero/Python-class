class SchedulePermission:
    def schedule_appointment(self):
        print("Scheduling appointment")

class BillingPermission:
    def process_billing(self):
        print("Billing patient")

class TreatingPermission:
    def treat_patient(self):
        print("Treating patient")

class Receptionist(SchedulePermission, BillingPermission):
    def schedule_appointment(self):
        print("Receptionist is: ")
        return super().schedule_appointment()
    
    def process_billing(self):
        print("Receptionist is: ")
        return super().process_billing()
class Doctor(TreatingPermission):
    def treat_patient(self):
        print("Doctor is: ")
        return super().treat_patient()
    
doctor = Doctor()
doctor.treat_patient()
receptionist = Receptionist()
receptionist.process_billing()
receptionist.schedule_appointment()