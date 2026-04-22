class Insurance:
    def __init__(self, customer, insurance_company):
        self.customer = customer
        self.insurance_company = insurance_company
    
    def choose_pcp(self):
        print(f"select your doctor dear {self.customer}")
class HMOInsurance(Insurance):
    def choose_pcp(self):
        print(f"HMO: A doctor must be selected")

class CarInsurance(Insurance):
    def choose_pcp(self):
        raise Exception("A car insurance is not a medical insurance")

def create_insurance(insurance):
    insurance.choose_pcp()

hmo = HMOInsurance("Alejandro", "BCBS")
create_insurance(hmo)
car_insurance = CarInsurance("Alejandro", "GEICO")
create_insurance(car_insurance)