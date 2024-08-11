import pymongo

url = 'mongodb+srv://bish150b:XWMy2SRuyRaDIAIK@cluster0.g4t9j0f.mongodb.net/'
client = pymongo.MongoClient(url)

db = client['adhd']

