"""
Author: Andrew Shepley
Contact: asheple2@une.edu.au
Source: Confluence
Methods
a) assign_boxes_to_classes
b) normalise_coordinates
c) confluence_nms - returns maxima scoring box, removes false positives using confluence - efficient
d) confluence - returns most confluent box, removes false positives using confluence - less efficient but better box
"""

from collections import defaultdict
import numpy as np

def assign_boxes_to_classes(bounding_boxes, classes, scores):
    """
    Parameters: 
       bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
       classes: list of class identifiers (int value, e.g. 1 = person)
       scores: list of class confidence scores (0.0-1.0)
    Returns:
       boxes_to_classes: defaultdict(list) containing mapping to bounding boxes and confidence scores to class
    """
    boxes_to_classes = defaultdict(list)
    for each_box, each_class, each_score in zip(bounding_boxes, classes, scores):
        if each_score >= 0.05:
            boxes_to_classes[each_class].append(np.array([each_box[0],each_box[1],each_box[2],each_box[3], each_score]))
    return boxes_to_classes

def normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y):
    """
    Parameters: 
       x1, y1, x2, y2: bounding box coordinates to normalise
       min_x,max_x,min_y,max_y: minimum and maximum bounding box values (min = 0, max = 1)
    Returns:
       Normalised bounding box coordinates (scaled between 0 and 1)
    """
    x1, y1, x2, y2 = (x1-min_x)/(max_x-min_x), (y1-min_y)/(max_y-min_y), (x2-min_x)/(max_x-min_x), (y2-min_y)/(max_y-min_y)
    return x1, y1, x2, y2

def confluence_nms(bounding_boxes,scores,classes,confluence_thr,gaussian,score_thr=0.05,sigma=0.5):  
    """
    Parameters:
       bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
       classes: list of class identifiers (int value, e.g. 1 = person)
       scores: list of class confidence scores (0.0-1.0)
       confluence_thr: value between 0 and 2, with optimum from 0.5-0.8
       gaussian: boolean switch to turn gaussian decaying of suboptimal bounding box confidence scores (setting to False results in suppression of suboptimal boxes)
       score_thr: class confidence score
       sigma: used in gaussian decaying. A smaller value causes harsher decaying.
    Returns:
       output: A dictionary mapping class identity to final retained boxes (and corresponding confidence scores)
    """
    
    class_mapping = assign_boxes_to_classes(bounding_boxes, classes, scores)
    output = {}
    for each_class in class_mapping:
        dets = np.array(class_mapping[each_class])
        retain = []
        while dets.size > 0:
            max_idx = np.argmax(dets[:, 4], axis=0)
            dets[[0, max_idx], :] = dets[[max_idx, 0], :]
            retain.append(dets[0, :])
            x1, y1, x2, y2 = dets[0, 0], dets[0, 1], dets[0, 2], dets[0, 3]
    
            min_x = np.minimum(x1, dets[1:, 0])
            min_y = np.minimum(y1, dets[1:, 1])
            max_x = np.maximum(x2, dets[1:, 2])   
            max_y = np.maximum(y2, dets[1:, 3])
    
            x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y)
            xx1, yy1, xx2, yy2 = normalise_coordinates(dets[1:, 0], dets[1:, 1], dets[1:, 2], dets[1:, 3],min_x,max_x,min_y,max_y)

            md_x1,md_x2,md_y1,md_y2 = abs(x1-xx1),abs(x2-xx2),abs(y1-yy1),abs(y2-yy2) 
            manhattan_distance = (md_x1+md_x2+md_y1+md_y2)

            weights = np.ones_like(manhattan_distance)

            if (gaussian == True):
                gaussian_weights = np.exp(-((1-manhattan_distance) * (1-manhattan_distance)) / sigma)
                weights[manhattan_distance<=confluence_thr]=gaussian_weights[manhattan_distance<=confluence_thr]
            else:
                weights[manhattan_distance<=confluence_thr]=manhattan_distance[manhattan_distance<=confluence_thr]

            dets[1:, 4] *= weights
            to_reprocess = np.where(dets[1:, 4] >= score_thr)[0]
            dets = dets[to_reprocess + 1, :]     
        output[each_class]=retain

    return output

def confluence(bounding_boxes,scores,classes,confluence_thr,gaussian,score_thr=0.05,sigma=0.5):
    """
    Parameters:
       bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
       classes: list of class identifiers (int value, e.g. 1 = person)
       scores: list of class confidence scores (0.0-1.0)
       confluence_thr: value between 0 and 2, with optimum from 0.5-0.8
       gaussian: boolean switch to turn gaussian decaying of suboptimal bounding box confidence scores (setting to False results in suppression of suboptimal boxes)
       score_thr: class confidence score
       sigma: used in gaussian decaying. A smaller value causes harsher decaying.
    Returns:
       output: A dictionary mapping class identity to final retained boxes (and corresponding confidence scores)
    """

    class_mapping = assign_boxes_to_classes(bounding_boxes, classes, scores)
    output = {}
    for each_class in class_mapping:
        dets = np.array(class_mapping[each_class])
        retain = []
        while dets.size > 0:
            confluence_scores,proximities = [],[]
            while len(confluence_scores)<np.size(dets,0):
                current_box = len(confluence_scores)
               
                x1, y1, x2, y2 = dets[current_box, 0], dets[current_box, 1], dets[current_box, 2], dets[current_box, 3]
                confidence_score = dets[current_box, 4]
                xx1,yy1,xx2,yy2,cconf = dets[np.arange(len(dets))!=current_box, 0],dets[np.arange(len(dets))!=current_box, 1],dets[np.arange(len(dets))!=current_box, 2],dets[np.arange(len(dets))!=current_box, 3],dets[np.arange(len(dets))!=current_box, 4]
                min_x,min_y,max_x,max_y = np.minimum(x1, xx1),np.minimum(y1, yy1),np.maximum(x2, xx2),np.maximum(y2, yy2)    
                x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y)
                xx1, yy1, xx2, yy2 = normalise_coordinates(xx1, yy1, xx2, yy2,min_x,max_x,min_y,max_y)

                hd_x1,hd_x2,vd_y1,vd_y2 = abs(x1-xx1),abs(x2-xx2),abs(y1-yy1),abs(y2-yy2)
                proximity = (hd_x1+hd_x2+vd_y1+vd_y2)
                all_proximities = np.ones_like(proximity)
                cconf_scores = np.zeros_like(cconf)

                all_proximities[proximity <= confluence_thr] = proximity[proximity <= confluence_thr]
                cconf_scores[proximity <= confluence_thr]=cconf[proximity <= confluence_thr]
                if(cconf_scores.size>0):
                    confluence_score = np.amax(cconf_scores)
                else:
                    confluence_score = confidence_score
                if(all_proximities.size>0):
                    proximity = (sum(all_proximities)/all_proximities.size)*(1-confidence_score)
                else:
                    proximity = sum(all_proximities)*(1-confidence_score)
                confluence_scores.append(confluence_score)
                proximities.append(proximity)
            
            conf = np.array(confluence_scores)
            prox = np.array(proximities)

            dets_temp = np.concatenate((dets, prox[:, None]), axis=1)
            dets_temp = np.concatenate((dets_temp, conf[:, None]), axis=1)
            min_idx = np.argmin(dets_temp[:, 5], axis=0)
            dets[[0, min_idx], :] = dets[[min_idx, 0], :]
            dets_temp[[0, min_idx], :] = dets_temp[[min_idx, 0], :]
            dets[0,4]=dets_temp[0,6]
            retain.append(dets[0, :])

            x1, y1, x2, y2 = dets[0, 0], dets[0, 1], dets[0, 2], dets[0, 3]
            min_x = np.minimum(x1, dets[1:, 0])
            min_y = np.minimum(y1, dets[1:, 1])
            max_x = np.maximum(x2, dets[1:, 2])   
            max_y = np.maximum(y2, dets[1:, 3])
    
            x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y)
            xx1, yy1, xx2, yy2 = normalise_coordinates(dets[1:, 0], dets[1:, 1], dets[1:, 2], dets[1:, 3],min_x,max_x,min_y,max_y)
            md_x1,md_x2,md_y1,md_y2 = abs(x1-xx1),abs(x2-xx2),abs(y1-yy1),abs(y2-yy2) 
            manhattan_distance = (md_x1+md_x2+md_y1+md_y2)
            weights = np.ones_like(manhattan_distance)

            if (gaussian == True):
                gaussian_weights = np.exp(-((1-manhattan_distance) * (1-manhattan_distance)) / sigma)
                weights[manhattan_distance<=confluence_thr]=gaussian_weights[manhattan_distance<=confluence_thr]
            else:
                weights[manhattan_distance<=confluence_thr]=manhattan_distance[manhattan_distance<=confluence_thr]

            dets[1:, 4] *= weights
            to_reprocess = np.where(dets[1:, 4] >= score_thr)[0]
            dets = dets[to_reprocess + 1, :]    
        output[each_class]=retain
    return output
