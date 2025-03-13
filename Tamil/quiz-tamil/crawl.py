import requests
import re
from bs4 import BeautifulSoup, NavigableString, Tag
import urllib.parse
import datetime
url1 = "https://quiz.tamil.help/riddle-/"
url = 	"url_to_crawl.txt"
page = open(url)

# page = requests.get(url)
#print(page.content)
soup = BeautifulSoup(page, "html.parser")
# print(soup)
link_hash = {}


# print(soup)

#print(soup.find("div", {"class": "td-module-meta-info"}).getText())
#print(text)
retreival_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
try:
	divs = soup.find_all("div", {"class": "section"})#.get_text(separator="\n", strip=True)
	# print(text)
	#print(text)
except:
	print("No text")
		
rows = []

# print(divs)

for section in divs:
	# Extract the question
	question = section.find("blockquote").get_text(strip=True) if section.find("blockquote") else ""

	# Extract all options
	options = [opt.get_text(strip=True) for opt in section.find_all("div", itemprop="suggestedAnswer")]

	# Extract the correct answer (acceptedAnswer)
	correct_answer = section.find("div", itemprop="suggestedAnswer acceptedAnswer")
	correct_answer_text = correct_answer.get_text(strip=True) if correct_answer else ""

	# Format as TSV row (Question, Options, Correct Answer)
	row = f"{question}\t{', '.join(options)}\t{correct_answer_text}"
	rows.append(row)

# Convert to TSV format
tsv_output = "\n".join(rows)
print(url1)
print(retreival_time)
print(tsv_output)
