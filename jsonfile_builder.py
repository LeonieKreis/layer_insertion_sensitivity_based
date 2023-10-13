import json

k1 = 0
k2 = 0
no_123 = 1
no_223 = 3


################# first training method #########################
# read json file 1
with open(f"results_data/Exp{k1}_{no_123}.json") as file:
    # a_temp = list(json.load(file).values())
    f1 = json.load(file)

# read json file 2
with open(f"results_data/Exp{k2}_{no_123}.json") as file:
    # a_temp = list(json.load(file).values())
    f2 = json.load(file)


####### second training method ###############################
# read json file 1
with open(f"results_data/Exp{k1}_{no_223}.json") as file:
    # a_temp = list(json.load(file).values())
    f3 = json.load(file)

# read json file 2
with open(f"results_data/Exp{k2}_{no_223}.json") as file:
    # a_temp = list(json.load(file).values())
    f4 = json.load(file)


# put dictionieries in one dictionary

dict1 = {}
dict2 = {}

name_new_json_file_1 = ""
name_new_json_file_2 = ""


# write new json file 1
with open(name_new_json_file_1, 'w') as file:
    json.dump(dict1, file)

# write new json file 2
with open(name_new_json_file_2, 'w') as file:
    json.dump(dict2, file)
