import json
import Levenshtein
import numpy as np

entity_json = json.load(open('./entity_old.json', "r+", encoding='utf-8'))

out_dict = {}

for k, v in entity_json.items():
    labels_dict = {}
    for entity in v:
        levenshtein_list = []
        for each_entity in v:
            levenshtein_list.append(Levenshtein.distance(entity, each_entity))
        
        sort_idx = (np.array(levenshtein_list).argsort()).tolist()
        tmp_idx = sort_idx[1:11]
        tmp_list = []
        for idx in tmp_idx:
            tmp_list.append(str(v[idx]))
        labels_dict[entity] = tmp_list
    
    out_dict[k] = labels_dict

    
with open('./entity.json', 'w+') as f:
    json.dump(out_dict, f, indent=2)
