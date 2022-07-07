import pprint
import requests

class Weathermap:
    
    def get(self,city):
        s_city = f"Moscow,RU"
        city_id = 524901
        appid = "f52be84f9e5c42ec8a683baf86c67c56"
        
        res = requests.get("http://api.openweathermap.org/data/2.5/forecast",
                           params={'id': city_id, 'units': 'metric', 'lang': 'ru', 'APPID': appid})

        data = res.json()
        forecast = []
        for i in data['list']:
            forecast.append({
            'date':i['dt_txt'],
            'temperature':f"+{i['main']['temp']}",
            'description':i['weather'][0]['description']
            })
        return forecast    
            
class Cityinfo:
    def __init__(self,city):
        self.city = city
        self._weather_forecast = Weathermap()  
        
    def weather_forecast(self):
        return self._weather_forecast.get(self.city) 
        

def _main():
    cityinfo = Cityinfo("Москва")
    forecast = cityinfo.weather_forecast()
    pprint.pprint(forecast)


if __name__ == "__main__":
    _main() 
    
# не только для москвы (ввод с клавы города), получение id
# чтобы не было много запросов
# исключения
# если класс не будет работать, то другой