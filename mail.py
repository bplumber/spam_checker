import imaplib, email
import imp
import re
import xml

CLEANR = re.compile(r'<[^>]+>') 
user = 'capscrap01@gmail.com'
password = 'Burhan@69'
imap_url = 'imap.gmail.com'


def cleanhtml(raw_html):
    clean = re.compile('<.*?>')
    return CLEANR.sub('', raw_html)
def get_body(msg):
    if msg.is_multipart():
        return get_body(msg.get_payload(0))
    else:
        return msg.get_payload(None, True)
def search(key, value, con):
    result, data = con.search(None, key, '"{}"'.format(value))
    return data
def get_emails(result_bytes):
    msgs = [] # all the email data are pushed inside an array
    for num in result_bytes[0].split():
        typ, data = con.fetch(num, '(RFC822)')
        msgs.append(data)
 
    return msgs
print("33")
con = imaplib.IMAP4_SSL(imap_url)
print("36")
con.login(user, password)
print("40")
con.select('Inbox')
print("43")
msgs = get_emails(search('FROM', 'b.plumber@somaiya.edu', con))
print("41")
n_data = []
for msg in msgs[::-1]:
    for sent in msg:
        if type(sent) is tuple:
            content = str(sent[1], 'utf-8')
            data = str(content)
            try:
                indexstart = data.find("ltr")
                data2 = data[indexstart + 5: len(data)]
                indexend = data2.find("</div>")
                n_data.append(data2[0: indexend])
            except UnicodeEncodeError as e:
                pass
for i in n_data:
    print(i)
print("--------------------------------------------------")
clean_data = []
for i in n_data:
    temp = cleanhtml(i)
    clean_data.append(temp)
for i in clean_data:
    print(i)
