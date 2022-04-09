import poplib
user_name = 'capscrap01@gmail.com' 
    
passwd = 'Burhan@69'
pop3_server_domain = 'pop3.gmail.com'
pop3_server_port = '995'
print("9")
# Connect to pop3 email server.
mail_box = poplib.POP3_SSL(pop3_server_domain, pop3_server_port) 
mail_box.user(user_name) 
mail_box.pass_(passwd)
print("14") 
# Get number of existing emails.
number_of_messages = len(mail_box.list()[1])
print("15")
# Loop in the all emails.
for i in range(number_of_messages):
    # Get one email.
    for msg in mail_box.retr(i+1)[1]:
        # Get the email from address. 
        fromm = msg.get('b.plumber@somaiya.edu')
        if(fromm.indexOf('b.plumber@somaiya.edu')>-1):
            print(msg)
        else:
            print('This message is not the one you want')
mail_box.quit()

