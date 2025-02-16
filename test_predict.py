import requests

def predict_sentiment(tweet):
    url = "https://aip7heroku-436f0a6aa765.herokuapp.com/predict"
    headers = {"Content-Type": "application/json"}
    payload = {
        "input": tweet
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    else:
        return response.json()

# Example usage
tweet = "I am very happy today"
prediction = predict_sentiment(tweet)
print(prediction)