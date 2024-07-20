import json 
import numpy as np 
from collections import defaultdict

def point_in_interval(point,interval):
    if point <= interval[1] and point >= interval[0]:
        return 1
    else:
        return 0 
    
def count_number_FN(false_negative_dict):
    ct=0
    for vidname in false_negative_dict.keys():
        ct= ct + len(false_negative_dict[vidname])
    return ct

def count_number_proposals(input_dict):
    ct = 0 
    for vid_name in input_dict.keys():
        ct+= len(input_dict[vid_name])
    return ct


def FN_finder(point_dict_path, props_dict):
    with open(point_dict_path, 'r') as pred_f:
        point_dict = json.load(pred_f) 
    false_negative_dict = {}
    for vidname in props_dict.keys():
        for curr_item in point_dict[vidname]:
            point_label, curr_point = curr_item
            flag = False
            for dict_item in props_dict[vidname]:
                s, e = dict_item["segment"]
                proposal_label = dict_item["label"]
                if proposal_label == point_label and point_in_interval(curr_point,[s, e]):
                    flag = True
            if not flag:
                if vidname not in false_negative_dict.keys():
                    false_negative_dict[vidname]= []
                false_negative_dict[vidname].append(curr_item)
    return false_negative_dict



def props_analysis(props_dict):
    dur_list = defaultdict(list)
    for vidname in props_dict.keys():
        for dict_item in props_dict[vidname]:
            s, e = dict_item["segment"]
            label = dict_item["label"]
            dur_list[label].append(e-s)                                                
    stat_dict={}
    for label in dur_list.keys():
        stat_dict[label]  = np.mean(np.array(dur_list[label]))
    return stat_dict



def train_filter_proposals(base_proposals_path,point_dict_path,output_path):
    with open(base_proposals_path, 'r') as pred_f:
        base_proposals = json.load(pred_f) 
    base_proposals = base_proposals["results"]
    with open(point_dict_path, 'r') as pred_f:
        point_dict = json.load(pred_f) 
    ##########################################################################################################
    proposals_dict = {}
    for vidname in base_proposals.keys():        
        video_point_dict = defaultdict(list)
        for item in point_dict[vidname]:
            video_point_dict[item[0]].append(item[1])            
        proposals_dict[vidname] = []
        for dict_item in base_proposals[vidname]:            
            points_in_proposal = [point for point in video_point_dict[dict_item["label"]] if point_in_interval(point,dict_item["segment"])==1]
            if len(points_in_proposal) ==1:
                proposals_dict[vidname].append({"label": dict_item["label"], "point":points_in_proposal[0], "score": dict_item["score"], "segment": dict_item["segment"]})       
            else: 
                for point_idx in range(len(points_in_proposal)):
                    lower_bound, upper_bound = dict_item["segment"]
                    curr_point = points_in_proposal[point_idx]
                    if point_idx < len(points_in_proposal)-1:
                        upper_bound = (curr_point + points_in_proposal[point_idx+1])/2
                    if point_idx >0:
                        lower_bound = (curr_point +points_in_proposal[point_idx-1])/2
                    proposals_dict[vidname].append({"label": dict_item["label"], "point":curr_point, "score": dict_item["score"], "segment": [lower_bound,upper_bound]})       
    stat_dict = props_analysis(proposals_dict)
    false_negative_dict = FN_finder(point_dict_path, proposals_dict)    
    ##########################################################################################################
    new_proposals= defaultdict(list)
    for vidname in false_negative_dict.keys():
        for point_item in false_negative_dict[vidname]:
            point_label, curr_point = point_item
            avg_duration  = stat_dict[point_label]
            for dict_item in base_proposals[vidname]:
                s, e = dict_item["segment"]
                highest_score = -1
                highest_score_candidate = None
                if point_label ==dict_item["label"] and point_in_interval(curr_point,dict_item["segment"]) and highest_score < dict_item["score"]:
                    if e-s> avg_duration:                        
                        s = max(curr_point - avg_duration/2 ,s)                   
                        e = min(curr_point + avg_duration/2,e)
                    highest_score = dict_item["score"]
                    highest_score_candidate = [[s,e], dict_item["score"], dict_item["label"]]
            if highest_score_candidate is not None:
                segment, score, proposal_label = highest_score_candidate
                new_proposals[vidname].append({"label": proposal_label, "score": score, "point":curr_point, "segment": segment})
            else:
                start_point = max(0, curr_point-avg_duration/2)
                end_point = curr_point+avg_duration/2
                new_proposals[vidname].append({"label": point_label, "score": 0,  "point":curr_point, "segment": [start_point,end_point]})
    ##########################################################################################################
    vid_list = set(proposals_dict).union(new_proposals)
    merged_proposals = {vidname: proposals_dict.get(vidname, []) + new_proposals.get(vidname, []) for vidname in vid_list}
    ##########################################################################################################
    output_dict = {}
    for vidname in merged_proposals.keys():
        output_dict[vidname] = []         
        for curr_item in point_dict[vidname]:
            point_label, curr_point = curr_item
            highest_score = -1
            union_proposal = None
            for dict_item in merged_proposals[vidname]:
                if dict_item["label"] == point_label and point_in_interval(curr_point,dict_item["segment"]) and dict_item["score"]>highest_score:
                    union_proposal = {"label": dict_item["label"], "point":curr_point,  "score": dict_item["score"], "segment": dict_item["segment"]}
                    highest_score =  dict_item["score"]
            if union_proposal is not None:
                output_dict[vidname].append(union_proposal)
    ##########################################################################################################
    final_res = {}
    final_res['version'] = 'VERSION 1.3'
    final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}
    final_res['results'] = output_dict 
    with open(output_path+"/PL_dict.json", 'w') as f:   
        json.dump(final_res, f)
        f.close()        
    ##########################################################################################################
    # false_negative_dict = FN_finder(point_dict_path, output_dict)
    # prop_count = count_number_proposals(output_dict)    
    # FN_count = count_number_FN(false_negative_dict)
    # print("Number of False Negatives: ",FN_count, " Number of Proposals: ",prop_count)


