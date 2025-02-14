import requests
import re
from bs4 import BeautifulSoup, NavigableString, Tag
import urllib.parse
import datetime
url = "https://www.freshtamil.com/2018/11/tamil-vidukathaigal-with-answers/"
#page = open(url)
page = requests.get("https://www.freshtamil.com/2018/11/tamil-vidukathaigal-with-answers/")
#print(page.content)
soup = BeautifulSoup(page.content, "html.parser")
link_hash = {}


# print(soup)

#print(soup.find("div", {"class": "td-module-meta-info"}).getText())
#print(text)
retreival_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
try:
	div = soup.find("div", {"class": "entry-content"})#.get_text(separator="\n", strip=True)
	# print(text)
	#print(text)
except:
	print("No text")
		
if div:
	
	lines = []
	for child in div.children:
		if isinstance(child, NavigableString):
			stripped = child.strip()
			if stripped:
				lines.append(stripped)
		elif isinstance(child, Tag):
			# Get text with newlines preserved within the tag's content
			text = child.get_text().strip()
			if text:
				lines.append(text)
	text = '\n'.join(lines)

text = re.sub(r'\n+', '\n', text)
print(url)
print(retreival_time)
print(text)
	#print(title_html)

