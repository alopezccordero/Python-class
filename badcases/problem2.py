class NotificationData:
    def __init__(self, date, message, receiver):
        self.date = date
        self.message = message
        self.receiver = receiver
class NotificationSenderMethod:
    def notify(self, notification, method):
        if method == "email":
            print(f"Sending notification via email to {notification.receiver}")
        if method == "phone":
            print(f"Sending notification via phone to {notification.receiver}")
        if method == "fax":
            print(f"Sending notification via fax to {notification.receiver}")

info = NotificationData("04,22,2026", "hello", "Dr Kim")
sender = NotificationSenderMethod()
sender.notify(info, "email")
sender.notify(info, "phone")
sender.notify(info, "fax")