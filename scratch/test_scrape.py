import re
from bs4 import BeautifulSoup

try:
    with open("scratch/lcel_page.html", "r", encoding="utf-8") as f:
        html = f.read()
except FileNotFoundError:
    import requests
    url = "https://docs.langchain.com/oss/python/langchain/overview"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)
    html = response.text

soup = BeautifulSoup(html, "lxml")
links = set()
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "oss/python" in href or href.startswith("/") or href.startswith("https://docs.langchain.com"):
        links.add(href)

print("Found links:")
for link in sorted(list(links)):
    # Let's clean up relative links
    if link.startswith("/"):
        full_link = f"https://docs.langchain.com{link}"
    else:
        full_link = link
    if "oss/python" in full_link:
        print(full_link)
