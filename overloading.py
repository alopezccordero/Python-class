def describe_pet(pet_name, **properties):
    print(pet_name)
    for key, value in properties.items():
        print(f"{key}: {value}")

describe_pet("Alejandrito", age=15, color="White")