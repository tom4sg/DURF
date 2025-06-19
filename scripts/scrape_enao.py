import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://everynoise.com/everynoise1d.cgi?scope=all"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)

if response.status_code != 200:
    print(f"Failed to retrieve ENAO, status code: {response.status_code}")
    exit(1)

soup = BeautifulSoup(response.text, 'html.parser')

genres = []

for link in soup.find_all('a'):
    href = link.get('href')
    text = link.text.strip()
    if href and 'engenremap-' in href:
        genres.append({
            'genre': text,
            'link': "https://everynoise.com/" + href
        })

df = pd.DataFrame(genres)
print(df.head())
df.to_csv("enao_genres.csv", index=False)
