class NotificationInfo:
    def __init__(self, date, message, receiver):
        self.receiver = receiver
        self.date = date
        self.message = message

class NotificationSender:
    def notify(self, notification, method):
        if method == "email":
            print(f"sending notification to {notification.receiver} via email")
        elif method == "phonenumber":
            print(f"sending notification to {notification.receiver} via phone number")

notification = NotificationInfo("04/22/2026", "This is the message", "Alejandro")
send = NotificationSender()
send.notify(notification, "email")
send.notify(notification, "phonenumber")