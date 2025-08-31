import requests

url = "https://legalsummarizer-138485278829.us-central1.run.app/summarize/"

files = {"file": open("companyPolicies.txt", "rb")}  # replace with your doc
response = requests.post(url, files=files)

print(response.text)
