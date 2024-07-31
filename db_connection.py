import pymongo

url = 'mongodb+srv://productionuser:Wueq3us8m3t5K4Wn@cluster0.g4t9j0f.mongodb.net/'
client = pymongo.MongoClient(url)

db = client['adhd']

