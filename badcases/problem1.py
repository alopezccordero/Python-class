class customer:
    def __init__(self, name, age, email, gender):
        self.name = name
        self.age = age
        self.email = email
        self.gender = gender
    def validate_email(self):
        if "@gmail.com" not in self.email:
            raise ValueError("Invalid email")
    def validate_gender(self):
        GENDERS = ['male', 'female']
        if self.gender not in GENDERS:
            raise ValueError("Invalid gender")
    def save_to_file(self):
        print(f"Saving to file")

def create_customer(customer):
    customer.validate_email()
    customer.validate_gender()
    customer.save_to_file()
p = customer("Alejandro", 20, "alejandro.lopezcordero01@gmail.com", "male")
create_customer(p)