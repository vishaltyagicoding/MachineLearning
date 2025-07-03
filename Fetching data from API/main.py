from tkinter.font import names

import requests
import pandas as pd

# TODO 1. Exercise API
# url = "https://exercisedb.p.rapidapi.com/exercises"
#
# querystring = {"limit":"10","offset":"0"}
#
# headers = {
# 	"x-rapidapi-key": "33ff0333c1mshba7cb2c31dce161p19c065jsn197f8cfdcef9",
# 	"x-rapidapi-host": "exercisedb.p.rapidapi.com"
# }
#
# response = requests.get(url, headers=headers, params=querystring)
#
# print(response.json())
#
# df = pd.DataFrame.from_dict(response.json())
# print(df.info())
#
# print(df.to_csv("exercise.csv",index=False))

# TODO 2. Job Search
import requests

url = "https://jsearch.p.rapidapi.com/search"

querystring = {"query":"developer jobs in chicago","page":"1","num_pages":"1","country":"us","date_posted":"all"}

headers = {
	"x-rapidapi-key": "33ff0333c1mshba7cb2c31dce161p19c065jsn197f8cfdcef9",
	"x-rapidapi-host": "jsearch.p.rapidapi.com"
}

params = {"query": "developer jobs", "num_pages": "20", "date_posted": "all"}

response = requests.get(url, headers=headers, params=params)

print(response.json())

df = pd.DataFrame(response.json()['data'])[['job_title', "employer_name", "job_publisher", "job_employment_type", "job_apply_link","job_city", "job_google_link"]]
print(df.info())
print(df)
print(df.to_csv("Job Search.csv",index=False))