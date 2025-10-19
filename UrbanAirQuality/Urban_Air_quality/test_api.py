import requests

API_KEY = "cd448a584154129d29996716538abc71"
city = "Nairobi"

url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url, timeout=10)

print("Status:", response.status_code)
print("Response:", response.json())
