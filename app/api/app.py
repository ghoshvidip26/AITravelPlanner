from flask import Flask, request, jsonify
import os
import re
from flask_cors import CORS
import collections
# from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests
import datetime
from datetime import date,datetime
from dateutil.parser import parse
import google.generativeai as genai
import chromadb
from chromadb import Documents,EmbeddingFunction,Embeddings
from google.api_core import retry
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
import spacy
from collections import defaultdict
nlp = spacy.load('en_core_web_sm')
import math
from collections import Counter
import time

load_dotenv()
collections.Iterable = collections.abc.Iterable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def callAPI(city): 
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('OPEN_WEATHER_API_KEY')}"
    response = requests.get(url).json()

    if response.get("cod") != 200:
        print("Error:", response.get("message", "Unknown error"))
        return None

    name = response.get("name", city)
    country = response["sys"].get("country", "")
    weather_desc = response["weather"][0].get("description", "").title()
    
    lat = response["coord"]["lat"]
    lon = response["coord"]["lon"]

    temp_k = response["main"]["temp"]
    feels_like_k = response["main"]["feels_like"]
    temp_c = round(temp_k - 273.15, 2)
    feels_like_c = round(feels_like_k - 273.15, 2)
    humidity = response["main"]["humidity"]
    wind_speed = response["wind"]["speed"]
    return response

def fetchCityDetails(keyword): 
    URL = "https://test.api.amadeus.com/v1/reference-data/locations"
    accessToken = obtainAccessToken()
    headers = {
        "Authorization": f"Bearer {accessToken}"
    }
    params = {
        "keyword": keyword,
        "subType": "CITY,AIRPORT"
    }
    
    response = requests.get(URL, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data. Status Code: {response.status_code}, Response: {response.text}")
        return None

class GeminiEmbeddingFunction(EmbeddingFunction): 
    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings: 
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task
            ),
        )
        return [e.values for e in response.embeddings]

def predictFutureTemp(city): 
    temp_map = {}
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={os.getenv('OPEN_WEATHER_API_KEY')}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        print("Status Code:", response.status_code)
        details = response.json().get("list")
        for forecast in details: 
            timestamp = forecast["dt"]
            converted_time = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            temp_map[converted_time] = forecast["main"]["temp"]
        
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
    return temp_map

def flightSearchAPI(source,destination,departureDate,returnDate,adults,nonStop): 
    search_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    accessToken = obtainAccessToken()
    headers = {"Authorization": f"Bearer {accessToken}"}
    params = {
        "originLocationCode": source,
        "destinationLocationCode": destination,
        "departureDate": departureDate,
        "returnDate": returnDate,
        "adults": adults,
        "nonStop": nonStop,
        "max": 10
    }
    response = requests.get(search_url, headers=headers, params=params).json()
    return response

def obtainAccessToken(): 
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": os.getenv('AMEDUS_API_KEY'),          
        "client_secret": os.getenv('AMEDUS_API_SECRET')
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers).json()
    return response.get("access_token")

def hotelAPI(cityCode): 
    url = f"https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city?cityCode={cityCode}"
    accessToken = obtainAccessToken()
    headers = {"Authorization": f"Bearer {accessToken}"}
    response = requests.get(url, headers=headers).json()
    return response

def airportAPI(cityName, countryCode): 
    url=f"https://test.api.amadeus.com/v1/reference-data/locations?subType=CITY,AIRPORT&keyword={cityName}&countryCode={countryCode}"
    accessToken = obtainAccessToken()
    headers = {"Authorization": f"Bearer {accessToken}"}
    response = requests.get(url, headers=headers).json()
    return response

def getIATACode(cityName, countryCode):
    data = airportAPI(cityName, countryCode)
    location = data.get("data")[0]
    if location.get("subType")=="CITY": 
        iataCode = location.get("address").get("cityCode")
    else: 
        iataCode = location.get("address").get("countryCode")
    return iataCode

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

def findDistanceFromAirport(cityCode, countryCode): 
    hotels = hotelAPI(cityCode)
    airport = airportAPI(cityCode,countryCode)

    airportGeoCode = airport.get("data")[0].get("geoCode")
    airport_coords = (airportGeoCode['latitude'], airportGeoCode['longitude'])

    tempMap = {}
    for hotel in hotels.get("data", []):
        name = hotel.get("name")
        geoCode = hotel.get("geoCode")
        if name and geoCode:
            tempMap[name] = geoCode
    distance_data = {}
    for name, geo in tempMap.items():
        hotel_coords = (geo['latitude'], geo['longitude'])
        d = distance(airport_coords, hotel_coords)
        print(f"{name}: {d:.2f} km from airport")
        distance_data[name]=d
    return distance_data        

def predictPrice(departureCity, arrivalCity, departureDate): 
    url = "https://booking-com18.p.rapidapi.com/flights/v2/min-price-oneway"
    querystring={
        "departId":departureCity,
        "arrivalId":arrivalCity,
        "departDate":departureDate
    }
    headers = {
        "x-rapidapi-key": "dc66c7ea2emsha6c7a2e618149d4p17da80jsn5c1e1cd8cb27",
        "x-rapidapi-host": "booking-com18.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

def extractDate(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == 'DATE': 
            print("Extracted DATE:", ent.text)
            extractedDate = parse(ent.text)
            formatted_date = extractedDate.strftime('%Y-%m-%d')
            return {"date": formatted_date}
        
def extractCity(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            print("Extracted GPE:", ent.text)
            return {"cityName": ent.text}
    iata_code_match = re.findall(r'\b[A-Z]{3}\b', query)
    if iata_code_match:
        iata_code = iata_code_match[0]
        country_match = re.findall(r'\b[A-Z]{2}\b', query)
        try: 
            APIResponse = airportAPI(iata_code, country_match[0])
            data = APIResponse.get("data")
            for location in data:
                address=location.get("address")
                return address
        except:
            print("Error in API call for IATA code:", iata_code)
            return None
        
    if len(query.strip().split()) == 1:
        print("Extracted City (Single word query):", query.strip())
        return {"cityName": query.strip()}
    return None

def fetchPointsOfInterest(lat,long,radius): 
    url = f"https://test.api.amadeus.com/v1/shopping/activities?longitude={long}&latitude={lat}&radius={radius}"
    accessToken = obtainAccessToken()
    headers = {"Authorization": f"Bearer {accessToken}"}
    response = requests.get(url, headers=headers).json()
    return response

def summarize_forecast(temp_map):
    daily_temps = defaultdict(list)
    for timestamp, temp in temp_map.items():
        date = timestamp.split(' ')[0]
        daily_temps[date].append(temp)

    forecast_summary = ""
    for day, temps in list(daily_temps.items())[:5]:  
        max_temp = round(max(temps), 2)
        min_temp = round(min(temps), 2)
        avg_temp = round(sum(temps) / len(temps), 2)
        
        forecast_summary += f"{day}: Average {avg_temp}°C, High {max_temp}°C, Low {min_temp}°C"

    return forecast_summary

def fetchNews(query, date_entry): 
    year, month, day = map(int, date_entry.split('-'))
    date_obj = date(year, month, day)
    cityName = extractCity(query).get("cityName")
    url = f"https://newsapi.org/v2/everything?q={cityName}&from={date_obj}&sortBy=publishedAt&apiKey=8f5255313e8b40eaaaec8567a3e3788b"
    
    print("URL:", url)  
    response = requests.get(url)
    return response.json().get("articles")

def sentimentAnalysis(text): 
    prompt = f"""What is the sentiment of the following sentence, which is delimited with triple backticks? Just give the sentiment in one word if it is positive, very positive, negative, very negative or neutral.
    Review text: ```{text}```
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response

def assess_safety(news_list):
    """ Maps sentiment analysis results to a count and displays safety assessment """
    sentiments = Counter()
    
    for index, news in enumerate(news_list):
        sentiment = sentimentAnalysis(news)
        sentiments[sentiment.strip()] += 1
        print(f"Recent news: {news}")
        print(f"\n --- Sentiment Analysis Summary ---\n{sentiments}")
        if (index + 1) % 15 == 0:
            print("⚠️ Rate limit reached, waiting for a minute...")
            time.sleep(60)  
            
    print("\n--- Final Sentiment Analysis Summary ---")
    for sentiment, count in sentiments.items():
        print(f"{sentiment}: {count}")
    
    print("\n--- Safety Assessment ---")
    if sentiments['Negative'] > sentiments['Positive']:
        print("❌ The place does not seem safe based on recent news.")
    elif sentiments['Neutral'] >= max(sentiments['Positive'], sentiments['Negative']):
        print("⚠️ The situation seems STABLE but take precautions.")
    else:
        print("✅ The place seems safe to visit!")

@app.route('/weather', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        query = data.get('newMessage')  
        
        city = extractCity(query)
        dateExtract = extractDate(query)
        print("Date:", dateExtract)
        cityName=city.get("cityName")
        if not cityName:
            return jsonify({"error": "City not found in the query. Please mention a valid city."}), 400
                    
        currentTemp = callAPI(cityName)
        if not currentTemp:
            return jsonify({"error": f"Could not retrieve current temperature for {cityName}."}), 500
        countryCode = currentTemp.get('sys').get('country')
        if not countryCode:
            return jsonify({"error": f"Could not determine country for {cityName}."}), 500
        cityCode = getIATACode(cityName, countryCode)
        distanceFromAirport = findDistanceFromAirport(cityCode, countryCode)
        if not cityCode:
            return jsonify({"error": f"Could not determine IATA code for {cityName}."}), 500

        temp_map = predictFutureTemp(cityName)
        if not temp_map:
            return jsonify({"error": f"Could not retrieve forecast data for {cityName}."}), 500

        forecast_summary = summarize_forecast(temp_map)
        hotelsList = hotelAPI(cityCode)
        hotelName = []
        newsList = fetchNews(query, dateExtract.get("date"))
        headlines = []
        for news in newsList:
            headlines.append(news.get("title"))
        hotel_data = hotelsList.get("data", [])
        for hotel in hotel_data:
            hotelName.append({
                "name": hotel.get("name"),
                "latitude": hotel.get("geoCode", {}).get("latitude"),
                "longitude": hotel.get("geoCode", {}).get("longitude")
            })

        documents = [
            f"Information about {cityName}: {fetchCityDetails(cityName)}\n",
            f"Weather in {cityName}: {currentTemp['weather'][0]['main']}, "
            f"Temperature: {round(currentTemp['main']['temp'] - 273, 2)}°C, "
            f"Feels like: {round(currentTemp['main']['feels_like'] - 273, 2)}°C, "
            f"Latitude: {currentTemp['coord']['lat']}, Longitude: {currentTemp['coord']['lon']}. "
            f"\nHere's the 5-day forecast for {cityName}: {forecast_summary}\n"
            f"Hotel information in city {cityName}: {hotelName}\n",
            f"Distance from airport to hotel in {cityName}: {distanceFromAirport} km\n",
            f"Current news articles related to {cityName} on {dateExtract}: {headlines}\n",
            f"Based on the recent news articles, the sentiment analysis indicates: {assess_safety(headlines)}\n",
        ]
        
        DB_NAME = "weatherdb"
        embed_fn = GeminiEmbeddingFunction()
        embed_fn.document_mode = True

        chroma_client = chromadb.Client()
        db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
        
        all_ids = db.get()['ids']
        if all_ids:
            db.delete(ids=all_ids)

        db.add(documents=documents, ids=[str(i) for i in range(len(documents))])
        embed_fn.document_mode = False
        
        # Query Chroma DB for relevant passage based on user input
        result = db.query(query_texts=[query], n_results=1)
        [all_passages] = result["documents"]
        
        query_oneline = query.replace("\n", " ")
        
        prompt = f"""
        You are a helpful and friendly assistant that answers questions based on the reference passage below using **plain text** without Markdown formatting. 
        When answering, please keep in mind that the person you're talking to may not be familiar with technical terms, 
        so break things down in a simple, conversational way. Feel free to provide context and background to make sure the 
        answer is thorough and clear. If the passage doesn't help with answering the question, you should clearly state that you don't have information about the requested city.

        QUESTION: {query_oneline}
        """
        
        for passage in all_passages: 
            passage_oneline = passage.replace("\n", " ")
            prompt += f"PASSAGE: {passage_oneline}\n"
        models = genai.GenerativeModel('gemini-2.0-flash-lite')
        answer = models.generate_content(prompt)

        if not answer or not answer.text:
            return jsonify({"error": "Error generating a response from the AI model."}), 500

        response_data = {
            "city": cityName or "Unknown City",
            "countryCode": countryCode or "Unknown Country",
            "currentWeather": currentTemp or {},
            "forecastSummary": forecast_summary or "No Forecast Available",
            "hotels": hotelName or [],
            "distanceFromAirport": distanceFromAirport or "N/A",
            "news": headlines or [],
            "sentimentAnalysis": assess_safety(headlines) or "No Analysis Available",
            "answer": answer.text if answer and answer.text else "No answer generated"
        }
        response = jsonify(response_data)
        return response

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True,port=3001)