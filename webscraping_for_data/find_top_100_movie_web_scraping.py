import requests
from bs4 import BeautifulSoup
import random

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...',
    'Mozilla/5.0 (X11; Linux x86_64)...'
]

headers = {'User-Agent': random.choice(user_agents)}

response = requests.get("https://web.archive.org/web/20200518073855/https://www.empireonline.com/movies/features/best"
                        "-movies-2/")


soup = BeautifulSoup(response.text, features="html.parser")


movies = soup.find_all(name="h3", class_="title")

for movie in movies:
    print(movie.getText())


