import yaml

with open("./apr13_Results/result_acm.yaml") as f:
    result = yaml.load(f, Loader=yaml.SafeLoader)
acc = -1
loss = 900
target_key = None
for key in result.keys():
    # print(key, result[key])
    if result[key]["bestScore"] > acc:
        target_key = key
        acc = result[key]["bestScore"]
    # if result[key]["acc"] > acc:
    #     target_key = key

print(target_key, result[target_key])
