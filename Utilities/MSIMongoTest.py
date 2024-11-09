from pymongo import MongoClient

# Open MongoDB
myclient = MongoClient("mongodb://localhost:27017/")
flares_db = myclient["Flares"]
flares_table = flares_db["Flares"]
flares_table.delete_many({})  # clear out whatever is in there

flares_table.insert_one({"sample": "test"})