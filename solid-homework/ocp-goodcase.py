class NotificationInfo:
    def __init__(self, date, message, receiver):
        self.receiver = receiver
        self.date = date
        self.message = message


class NotificationMethod:
    def send(self, notification):
        pass


class EmailNotification(NotificationMethod):
    def __init__(self, email):
        self.email = email

    def send(self, notification):
        print(f"Sending notification to {notification.receiver} via email: {self.email}")


class PhoneNotification(NotificationMethod):
    def __init__(self, phone):
        self.phone = phone

    def send(self, notification):
        print(f"Sending notification to {notification.receiver} via phone: {self.phone}")


class NotificationSender:
    def notify(self, notification, method):
        method.send(notification)



sender = NotificationSender()

email = EmailNotification("alejandro.lopezcordero01@utrgv.edu")
phone = PhoneNotification("+19561111111")

info = NotificationInfo("04/22/2026", "This is the new notification", "Alejandro")

sender.notify(info, email)
sender.notify(info, phone)