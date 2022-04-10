import poplib
from email.parser import Parser
user_name = 'capscrap01@gmail.com' 
passwd = 'Burhan@69'
pop3_server_domain = 'pop.gmail.com'
pop3_server_port = '995'
mail_box = poplib.POP3_SSL(pop3_server_domain, pop3_server_port)
print(mail_box)
mail_box.set_debuglevel(1)
pop3_server_welcome_msg = mail_box.getwelcome().decode('utf-8')
mail_box.user(user_name)
mail_box.pass_(passwd)
print(mail_box.list())
resp, mails, octets = mail_box.list()

print(octets)
index = len(mails)
print(index)
clean_data = []
for i in range(5):
    try:
        resp, lines, octets = mail_box.retr(i)
        msg_content = b'\r\n'.join(lines).decode('utf-8')
        msg = Parser().parsestr(msg_content)
        email_subject = msg.get('Subject')
        clean_data.append(email_subject)
    except:
        pass
print(clean_data)
mail_box.quit()