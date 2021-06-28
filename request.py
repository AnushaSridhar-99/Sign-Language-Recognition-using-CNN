import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'language' = 'en', 'sequence' = 'Sequence-2'})

print(r.json())