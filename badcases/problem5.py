class MessageSender:#low level module
    def send(self, sender_name, receiver, email):
        print(f"sending message from {sender_name} to {receiver} to email {email}")

class Message:
    def __init__(self, body, sender_name, receiver, email):
        self.body = body
        self.sender_name = sender_name
        self.receiver = receiver
        self.email = email
        self.sender = MessageSender()

    def send_email(self):
        print("Sending email")
        self.sender.send(self.sender_name, self.receiver, self.email)

message = Message("this is the body", "Alejandro", "Carlos", "carlos@test.com")
message.send_email()