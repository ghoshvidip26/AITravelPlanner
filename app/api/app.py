from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import collections
from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests
from datetime import datetime
import chromadb
from chromadb import Documents,EmbeddingFunction,Embeddings
from google.api_core import retry
from google.genai import types
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
import spacy
from collections import defaultdict
nlp = spacy.load('en_core_web_sm')
from datetime import datetime
from IPython.display import display, Markdown, Latex

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPEN_WEATHER_API_KEY = os.getenv('OPEN_WEATHER_API_KEY')
collections.Iterable = collections.abc.Iterable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type"])

client = genai.Client(api_key=GOOGLE_API_KEY)

def callAPI(city): 
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPEN_WEATHER_API_KEY}"
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

    sunrise = datetime.fromtimestamp(response["sys"]["sunrise"]).strftime('%H:%M:%S')
    sunset = datetime.fromtimestamp(response["sys"]["sunset"]).strftime('%H:%M:%S')

    return response
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
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPEN_WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        print("Status Code:", response.status_code)
        details = response.json().get("list")
        for date in details: 
            timestamp = date["dt"]
            converted_time = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            temp_map[converted_time] = date["main"]["temp"]
        
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
    return temp_map

def extractCity(query): 
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            print(ent.text)
            return ent.text
    return None

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
        
        forecast_summary += (
            f"\n{day}: Average {avg_temp}°C, High {max_temp}°C, Low {min_temp}°C\n"
        )
    return "\n".join(forecast_summary.strip())

@app.route('/weather', methods=['POST'])
def predict():   
    data = request.get_json()
    query = data.get('newMessage')  
    city = extractCity(query)
    if not city:
        return jsonify({"error": "City not found in the query."}), 400
    print("City found: ",city)
    currentTemp = callAPI(city)  
    temp_map = predictFutureTemp(city)
    forecast_summary = summarize_forecast(temp_map)
    
    documents = [
        f"Weather in {city}: {currentTemp['weather'][0]['main']}, "
        f"Temperature: {round(currentTemp['main']['temp'] - 273, 2)}°C, "
        f"Feels like: {round(currentTemp['main']['feels_like'] - 273, 2)}°C, "
        f"Latitude: {currentTemp['coord']['lat']}, Longitude: {currentTemp['coord']['lon']}. "
        f"\nHere's the 5-day forecast for {city}: {forecast_summary}\n"
    ]
    DB_NAME="weatherdb"
    embed_fn=GeminiEmbeddingFunction()
    embed_fn.document_mode=True

    chroma_client=chromadb.Client()
    db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
    db.add(documents=documents, ids=[str(i) for i in range(len(documents))])
    embed_fn.document_mode=False
    
    result=db.query(query_texts=[query],n_results=1)
    [all_passages]=result["documents"]
    query_oneline = query.replace("\n", " ")
    
    prompt = f"""
    You are a helpful and friendly assistant that answers questions based on the reference passage below using **plain text** without Markdown formatting. 
    When answering, please keep in mind that the person you're talking to may not be familiar with technical terms, 
    so break things down in a simple, conversational way. Feel free to provide context and background to make sure the 
    answer is thorough and clear. If the passage doesn't help with answering the question, you can leave it out.

    QUESTION: {query_oneline}
    """
    
    for passage in all_passages: 
        passage_oneline=passage.replace("\n"," ")
        prompt += f"PASSAGE: {passage_oneline}\n"
    answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)
    print("Upcoming temperatures:", forecast_summary)
    return jsonify({
        "currentTemperature": currentTemp,
        "temperatureInUpcomingDays": temp_map,
        "response": answer.text
    })

if __name__ == '__main__':
    app.run(debug=True,port=3000)