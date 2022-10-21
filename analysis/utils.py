import os
import requests
import json
from collections import defaultdict 
from typing import Dict, Tuple, Any 

import pandas as pd

def get_metrics(url):
    if os.path.isfile(url):
        print(f'load metrics from local file: {url}')
        return json.load(open(url))
    # get link from: download button in W&B dashboard -> right click -> copy link 
    print(f'load metrics via get request: {url}')
    r = requests.get(url, allow_redirects=True)
    print(r)
    content = json.loads(r.content)
    return content

def normalize_keys(metrics):
    metrics = {k.replace("__", "_").replace("/fscore", "/f1"):v for k, v in metrics.items()}
    for agg in ["mean", "std", "median"]:
        metrics = {k.replace(f'_{agg}', f'/{agg}'): v for k, v in metrics.items()}
    metrics = {k[1:] if k.startswith("_") else k: v for k, v in metrics.items()} 
    return metrics

def filter_prefix(metrics, prefix, join=None):
    if join is not None:
        prefix = prefix + join
    return {k.replace(prefix, ""):v for k, v in metrics.items() if k.startswith(prefix)}

def filter_suffix(metrics, suffix, join=None):
    if join is not None:
        suffix = join + suffix
    return {k.replace(suffix, ""):v for k, v in metrics.items() if k.endswith(suffix)}

def to_nested_dict(d: Dict[Tuple, Any]) -> Dict:
    res = defaultdict(dict)
    for k, v in d.items():
        if len(k) > 1:
            res[k[0]].update({k[1:]: v})
        else:
            res[k[0]] = v
    return {k: to_nested_dict(v) if isinstance(v, dict) else v for k, v in res.items()}

def l2_to_df(l2_dict):
    res = defaultdict(dict)
    for k, v in l2_dict.items():
        res[k[0]][k[1]] = v
    return pd.DataFrame(res)

def reorganize_metrics(metrics, show=True):
    metrics_selected = filter_prefix(metrics, prefix="best_validation", join="_")
    if len(metrics_selected) == 0:
        metrics_selected = metrics

    metrics_selected = {tuple(k.split("/")): v for k, v in metrics_selected.items()}
    #display(metrics_selected)
    #display(to_nested_dict(metrics_selected))
    
    # iterate entries by number of key parts
    l = 1
    res = {}
    while len(metrics_selected) > 0:
        current = {}
        for k in list(metrics_selected):
            if len(k) == l:
                current[k] = metrics_selected.pop(k)
        as_nested_dict = to_nested_dict(current)
        #display(as_nested_dict)

        if len(current) > 0:
            if l == 1:
                res[l] = pd.Series(current)
            elif l == 2:
                res[l] = l2_to_df(current)
                #res[l] = pd.DataFrame(as_nested_dict)#
            elif l == 3:
                res[l] = {k: pd.DataFrame(v) for k, v in as_nested_dict.items()}
                #tmp = []
                #for k, v in as_nested_dict.items():
                #    df = pd.DataFrame(v).T
                #    df.columns = pd.MultiIndex.from_product([[k], df.columns])
                #    tmp.append(df)   
                #res[l] = pd.concat(tmp, axis=1)
            elif l == 4:
                res[l] = {k1: {k: pd.DataFrame(v) for k, v in v1.items()} for k1, v1 in as_nested_dict.items()}
            else:
                res[l] = as_nested_dict
        l += 1
        #break
    
    if show:
        for k, v in res.items():
            print(k)
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    print(k1)
                    display(v1)
                    #print(json.dumps(v1, indent=2))

            else:
                display(v)
    return res