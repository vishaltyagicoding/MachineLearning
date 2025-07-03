import time

import requests
from bs4 import BeautifulSoup
import random
import pandas as pd
from itertools import zip_longest
companies_name_ = []
ratings_ = []
rating_counts_ = []
interLinkings_ = []
salarys_ = []
about_companies_ = []
links_ = []

def add_data(page_num):
    url = f"https://www.ambitionbox.com/list-of-companies?page={page_num}"
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...',
        'Mozilla/5.0 (X11; Linux x86_64)...'
    ]

    headers = {'User-Agent': random.choice(user_agents)}


    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, features="html.parser")


    companies_name = soup.find_all(name="h2", class_="companyCardWrapper__companyName")
    ratings = soup.find_all(name="div", class_="rating_text")
    rating_counts = soup.find_all(name="span", class_="companyCardWrapper__companyRatingCount")
    interLinkings = soup.find_all(name="span", class_="companyCardWrapper__interLinking")
    salarys = soup.find_all(name="span", class_="companyCardWrapper__ActionCount")
    about_companies = soup.find_all(name="span", class_="companyCardWrapper__interLinking")


    for name in companies_name:
        n = name.get_text().strip()
        companies_name_.append(n)
        links_.append(f"https://www.ambitionbox.com/overview/{n}-overview")

    for rating in ratings:
        ratings_.append(rating.get_text().strip())

    for rating_count in rating_counts:
        rating_counts_.append(rating_count.get_text().strip())

    for interLinking in interLinkings:
        interLinkings_.append(interLinking.text.strip())


    for i in range(1, len(salarys), 6):
        salarys_.append(salarys[i].get_text().strip())

    for about in about_companies:
         about_companies_.append(about.get_text().strip())

for x in range(1, 501):
    print(x)
    add_data(x)
    # time.sleep(1)
# companies_name_, ratings_, rating_counts_, interLinkings_, salarys_ = zip(*zip_longest(companies_name_, ratings_, rating_counts_, interLinkings_,salarys_, fillvalue=None))
df = pd.DataFrame(
    {
        "Company Name": companies_name_,
        "Company Rating": ratings_,
        "Company Review": rating_counts_,
        "Sector": interLinkings_,
        "Salary": salarys_,
        "About Company": about_companies_,
        "Links": links_
    },
)

df.to_csv("companies.csv", index=False)

print(df)

# add new row in the exiting csv file

# new_row = pd.DataFrame(
#     {
#         "Company Name": companies_name_,
#         "Company Rating": ratings_,
#         "Company Review": rating_counts_,
#         "Sector": interLinkings_,
#         "Salary": salarys_,
#         "About Company": about_companies_,
#         "Links": links_
#     },
# )
# df = pd.read_csv("companies.csv")
# df = pd.concat([df, new_row], ignore_index=True)
# df.to_csv("companies.csv", index=False)


