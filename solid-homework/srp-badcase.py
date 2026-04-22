class Patient:
    def __init__(self, name, age, gender, email):
        CHOICES = [
            'male',
            'female'
        ]
        if "gmail.com" not in email:
            raise ValueError("Please enter a valid email")
        if gender not in CHOICES:
            raise ValueError("Gender must be 'male' or 'female'")
        self.name = name
        self.age = age
        self.gender = gender
        self.email = email

class PatientRecord:
    def save(self, patient):
        print(f"saving {patient.name} to records")

class AppointmentNotifier:
    def appointment_reminder(self, patient):
        print(f"Sending email to {patient.email}. patient: {patient.name}. Appointment scheduled.")