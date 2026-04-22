class Insurance:
    def __init__(self, patient_name, insurance_company):
        self.patient_name = patient_name
        self.insurance_company = insurance_company

        print(patient_name)
        print(insurance_company)


class PCPRequiredIns(Insurance):
    def __init__(self, patient_name, insurance_company, pcp):
        super().__init__(patient_name, insurance_company)
        self.pcp = pcp
    def choose_pcp(self):
        print(f"Choosing dr {self.pcp} as a primary care provider for patient {self.patient_name}\n")
        



class HMOInsurance(PCPRequiredIns):
    def choose_pcp(self):
        print("HMO: primary care provider required - you must choose one ")
        super().choose_pcp()

class PPOInsurance(Insurance):
    def open_network(self):
        print("PPO: open network - you can choose to go to any doctor")


def process_insurance(insurance):
    if isinstance(insurance, PCPRequiredIns):
        insurance.choose_pcp()
    else:
        insurance.open_network()

hmo_insurance = HMOInsurance("Alejandro", "BCBS", "Dr Kim")
process_insurance(hmo_insurance)

ppo_insurance = PPOInsurance("Carlos", "UHC")


process_insurance(ppo_insurance)