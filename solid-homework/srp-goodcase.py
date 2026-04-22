class Patient:
    def __init__(self, name, age, gender, email):

        self.name = name
        self.age = age
        self.gender = gender
        self.email = email

class PatientValidation:
    GENDERS = ['male', 'female']

    def validate(self, patient):
        if "@gmail.com" not in patient.email:
            raise ValueError("Invalid email")
        if patient.gender not in self.GENDERS:
            raise ValueError("Gender must be 'male' or 'female'")

class PatientRecord:
    def save(self, patient):
        file = open(f"{patient.name} - record.txt", 'w')
        file.write(
            f"Name: {patient.name}\n"
            f"Age: {patient.age}\n"
            f"Gender: {patient.gender}\n"
            f"Email: {patient.email}\n"
        )
        file.close()
        print(f"saving {patient.name} to records")

class AppointmentNotifier:
    def appointment_reminder(self, patient):
        print(f"Sending email to {patient.email}. patient: {patient.name}. Appointment scheduled.")


patient = Patient("alex", 20, "male", "alejandro.lopezcordero01@gmail.com")

validator = PatientValidation()
validator.validate(patient)

record = PatientRecord()
record.save(patient)

Notification = AppointmentNotifier()

Notification.appointment_reminder(patient)

