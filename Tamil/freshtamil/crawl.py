import requests
import re
from bs4 import BeautifulSoup, NavigableString, Tag
import urllib.parse
import datetime
url = "https://www.freshtamil.com/2020/04/tamil-vidukathaigal-in-tamil/"
#page = open(url)
page = requests.get(url)
#print(page.content)
soup = BeautifulSoup(page.content, "html.parser")
link_hash = {}


# print(soup)

#print(soup.find("div", {"class": "td-module-meta-info"}).getText())
#print(text)
retreival_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
try:
	table = soup.find("table")#.get_text(separator="\n", strip=True)
	# print(text)
	#print(text)
except:
	print("No text")
		
rows = []
for row in table.find_all("tr"):
    cells = row.find_all(["th", "td"])
    row_text = "\t".join(cell.get_text(strip=True) for cell in cells)  # Join cells with tab
    rows.append(row_text)

# Convert to tab-separated format
tsv_output = "\n".join(rows)

print(tsv_output)
	#print(title_html)

