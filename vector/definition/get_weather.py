import requests
import os
import json
api_key = os.getenv("GEOCODING_API")

def get_weather_forecast(address, api_key):

    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}&language=ko"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        lat = location['lat']
        lng = location['lng']
    else:
        raise ValueError("위치를 찾을 수 없습니다.")

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly=temperature_2m,precipitation,cloudcover,windspeed_10m,uv_index,apparent_temperature,precipitation_probability,relative_humidity_2m&forecast_hours=24&timezone=Asia/Seoul"
    result = requests.get(url).json()
    weather_text = ""
    for i in range(len(result['hourly']['time'])):
        t = result['hourly']['time'][i]
        tmp = result['hourly']['temperature_2m'][i]       # °C, 기온
        rain = result['hourly']['precipitation'][i]       # mm, 강수량
        rain_prob = result['hourly']['precipitation_probability'][i]  # %, 강수확률
        cloud = result['hourly']['cloudcover'][i]         # %, 구름양
        wind = result['hourly']['windspeed_10m'][i]       # km/h, 풍속
        uv = result['hourly']['uv_index'][i]             # UV 지수
        feel_temp = result['hourly']['apparent_temperature'][i]  # °C, 체감온도
        humidity = result['hourly']['relative_humidity_2m'][i]  # %, 상대습도

        weather_text += f"{t}: 온도 {tmp}℃ (체감 {feel_temp}℃), 강수량 {rain}mm (확률 {rain_prob}%), 구름 {cloud}%, 풍속 {wind}km/h, UV {uv}, 습도 {humidity}%\n"
    return weather_text
