class Insurance:
    def __init__(self, patient_name, insurance_company):
        self.patient_name = patient_name
        self.insurance_company = insurance_company
    
    def choose_pcp(self):
        print(f"Choosing primary care provideR: ")



class HMOInsurance(Insurance):
    def choose_pcp(self):
        print("HMO: primary care provider required - you must choose one ")

class PPOInsurance(Insurance):
    def choose_pcp(self):
         raise Exception("PPo does not require a primary care provider")

def process_insurance(insurance):
        insurance.choose_pcp()

hmo_insurance = HMOInsurance("Alejandro", "BCBS", "Dr Kim")
ppo_insurance = PPOInsurance("Carlos", "UHC")

process_insurance(hmo_insurance)

process_insurance(ppo_insurance)