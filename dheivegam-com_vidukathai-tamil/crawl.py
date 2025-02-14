import requests
import re
from bs4 import BeautifulSoup
import urllib.parse
import datetime
url = "https://dheivegam.com/vidukathai-tamil"
#page = open(url)
page = requests.get("https://dheivegam.com/vidukathai-tamil/")
#print(page.content)
soup = BeautifulSoup(page.content, "html.parser")
link_hash = {}


print(soup)

#print(soup.find("div", {"class": "td-module-meta-info"}).getText())
#print(text)
retreival_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
try:
	text = soup.find("div", {"class": "td-pb-row"}).getText()
	print(text)
	#print(text)
except:
	print("No text")
		
text = re.sub(r'\uFFFD', "-", text) #replacement character
text = re.sub(r'\u000C', "", text)	#formfeed
text = re.sub(r'\u00A0'," ", text)    #convert nbsp space to normal space
text = re.sub(r'\t', " ", text)     #tab to space
text = re.sub(r'^ *',"", text)
text = re.sub(r' *$',"", text)
text = re.sub(r' +', " ", text)
text = re.sub(r'\n+', "", text)
#print(text)
text = re.sub(r'Google map.*$\n', '', text)
#print(text)

fp = open(filename + ".txt", "w")
fp.write(current_link + "\n" + retreival_time + "\n" + text)
fp.close()

	#print(title_html)

