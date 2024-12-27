import json
import numpy as np 
from random import randint 
import cv2 


all_data_directory = "All_data/"

ann_file_train = all_data_directory + 'Annotations_vcoco/train_annotations.json'
ann_file_val = all_data_directory + 'Annotations_vcoco/val_annotations.json'
ann_file_test = all_data_directory + 'Annotations_vcoco/test_annotations.json'

with open(ann_file_train) as f:
    ANNOTATIONS_TRAIN = json.load(f)
    
with open(ann_file_val) as f:
    ANNOTATIONS_VAL = json.load(f) 
    
with open(ann_file_test) as f:
    ANNOTATIONS_TEST = json.load(f)
    
OBJ_path_train = all_data_directory + 'Object_Detections_vcoco/train/'
OBJ_path_test = all_data_directory + 'Object_Detections_vcoco/val/'

VERB_TO_ID = {
    'carry': 0,
    'catch': 1,
    'cut_instr':2,
    'cut_obj': 3,
    'drink': 4,
    'eat_instr':5,
    'eat_obj': 6,
    'hit_instr':7,
    'hit_obj': 8,
    'hold': 9,
    'jump': 10,
    'kick': 11,
    'lay': 12,
    'look': 13,
    'point': 14,
    'read': 15,
    'ride': 16,
    'run': 17,
    'sit': 18,
    'skateboard': 19,
    'ski': 20,
    'smile': 21,
    'snowboard': 22,
    'stand': 23,
    'surf': 24,
    'talk_on_phone': 25,
    'throw': 26,
    'walk': 27,
    'work_on_computer': 28
}

MATCHING_IOU = 0.5
NUMBER_OF_VERBS = 29

def get_detections(segment_key, flag):
    
    """based on threshold score values, for score person and score obj in object detctions, we distinct the object detections. This includes actual co-ordinate of the person and object in the images.
    
    segment_key = train / test / val.
    flag = image_no.

    Returns:
        d_p_boxes = all the presons bbx in the image -> actual image co-ordinates
        d_o_boxes = all the objects bbx in the image -> actual image co-ordinates
        scores_p = person scores in the image.
        scores_o = object scores in the image
        class_id_persons = class_id for the person boxes in image.
        class_id_objects = class_id for the object boxes in image.
        annotation = cleaned up annotation of the form [{'person_box': [0.96, 1.07, 352., 145], 
          'hois': [{'verb': 'cut_obj', 'obj_box': [117.61, 175.46, 522.51, 332.6]}, {'verb': 'hold', 'obj_box': [163.17, 50.3, 231.19, 116]}]}].
        img_shape = shape of this image (W, H)
    """
    
    SCORE_PER = 0.6
    SCORE_OBJ = 0.3
    select_threshold=2000000
    if flag == 'train':
        annotation = ANNOTATIONS_TRAIN[str(segment_key)]
        cur_obj_paths = OBJ_path_train + "COCO_train2014_%.12i.json" % (segment_key)
        
    elif flag == 'test':
        annotation = ANNOTATIONS_TEST[str(segment_key)]
        cur_obj_paths = OBJ_path_test + "COCO_val2014_%.12i.json" % (segment_key)
        
    elif flag == 'val':
        annotation = ANNOTATIONS_VAL[str(segment_key)]
        cur_obj_paths = OBJ_path_train + "COCO_train2014_%.12i.json" % (segment_key)
    
    annotation = clean_up_annotation(annotation)
    
    with open(cur_obj_paths) as f:
        detections = json.load(f)
        
    img_H = detections['H']
    img_W = detections['W']
    img_shape = [img_W, img_H]
    persons_d, objects_d = analyze_detections(detections, SCORE_PER, SCORE_OBJ)
    d_p_boxes, scores_p, class_id_persons = get_boxes_det(persons_d, img_H, img_W)
    d_o_boxes, scores_o, class_id_objects = get_boxes_det(objects_d, img_H, img_W)
    
    if len(d_p_boxes)>select_threshold:
        d_p_boxes,scores_p ,class_id_persons= d_p_boxes[0:select_threshold],scores_p[0:select_threshold],class_id_persons[0:select_threshold]
        
    if len(d_o_boxes)>select_threshold-1:
        d_o_boxes,scores_o,class_id_objects= d_o_boxes[0:select_threshold-1],scores_o[0:select_threshold-1],class_id_objects[0:select_threshold-1]
        
    return d_p_boxes, d_o_boxes, scores_p, scores_o, class_id_persons, class_id_objects, annotation, img_shape




def analyze_detections(detections, SCORE_PER, SCORE_OBJ):
    
    """gives all the person predicitons and the objects in an image seperately.
       based on high scores.

    Returns:
        two lists, one is the persons detected in an image, the other is the objects detected in an image.
    """
    
    persons = []
    objects = []
    
    for det in detections['detections']:
        if det['class_str'] == 'person':
            if det['score'] >= SCORE_PER:
                persons.append(det)
        
        else:
            if det['score'] >= SCORE_OBJ:
                objects.append(det)
                
    return persons, objects


def get_boxes_det(dets, img_H, img_W):
    
    """Gives the distinct boxes, scores and classes present in the detection recieved
       Gives us the actual co ordinates in the image for the persons and objects. 

    Args:
        dets (_type_): [{'class_str': 'tie','score': 0.063, 'class_no': 28, 'box_coords': [0.06, 0.109, 0.847, 0.535]}]
        img_H (_type_): height
        img_W (_type_): width

    Returns:
        type: boxes, scores, class_no. present in the given detection (Person / Object).
    """
    
    boxes = []
    scores = []
    class_no = []
    
    for det in dets:
        top, left, bottom, right = det['box_coords']
        scores.append(det['score'])
        class_no.append(det['class_no'])
        left, top, right, bottom = left* img_W, top*img_H, right*img_W, bottom*img_H
        boxes.append([left, top, right, bottom])
    
    return boxes, scores, class_no




def clean_up_annotation(annotation):
    
    """
    Recieves the main annotation structure that is present in the main dataset
       for an image, and converts it into more explainable and easily readable annotation
       for verb with no objectm, the object bbx remains []

    Args:
        annotation (_type_): 
        '106497': [{'person_bbx': [0.96, 1.07, 352., 145],
                    'Verbs': 'cut_obj',
                    'object': {'obj_bbx': [117.61, 175.46, 522.51, 332.6]}},
                   {'person_bbx': [0.96, 1.07, 3525, 145],
                    'Verbs': 'hold',
                    'object': {'obj_bbx': [163.17, 50.3, 231.19, 116]}}]

    Returns:
        _type_: 
        [{'person_box': [0.96, 1.07, 352., 145], 
          'hois': [{'verb': 'cut_obj', 'obj_box': [117.61, 175.46, 522.51, 332.6]}, {'verb': 'hold', 'obj_box': [163.17, 50.3, 231.19, 116]}]}]  
    """
    
    persons_dict = {}
    
    for hoi in annotation:
        box = hoi['person_bbx']
        box = [int(coord) for coord in box]
        dkey = tuple(box)
        objects = hoi['object']
        
        if len(objects['obj_bbx']) == 0:
            cur_oi = {
                'verb': hoi['Verbs'],
                'obj_box': []
            }
        else:
            cur_oi = {
                'verb': hoi['Verbs'],
                'obj_box': [int(coord) for coord in objects['obj_bbx']]
            }
        if dkey not in persons_dict:
            persons_dict[dkey] = {'person_box': box, 'hois':[cur_oi]}
        else:
            persons_dict[dkey]['hois'].append(cur_oi)
    pers_list = []
    
    for dkey in persons_dict.keys():
        pers_list.append(persons_dict[dkey])
    
    return pers_list





def get_compact_detections(segment_key, flag):
    
    """This thing is required for building attention maps.
       here, we get the numpy array version for the person bboxes. 
       here it is based on 0 to 1 value. not the image co-ordinate value.
       in objects_np, there is an added co-ordinate , this is for no object involved verbs.
       and the co-ordinate is 0, 0, 0, 0.
       as person class is 0, for this specific object, class is given 0.
       it can be seen in class_id_objects, 1st val is 1.

    Returns:
        _type_: _description_
    """
    
    d_p_boxes, d_o_boxes, scores_p, scores_o, class_id_persons, class_id_objects, annotation, img_shape = get_detections(segment_key, flag)
    
    img_W, img_H = img_shape[0], img_shape[1]
    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    persons_np = np.zeros([no_person_dets, 4], np.float32)
    objects_np = np.zeros([no_object_dets+1, 4], np.float32)
    class_id_objects.insert(0, 1)
    if no_person_dets != 0:
        persons_np = np.array(d_p_boxes, np.float32)
    objects_np = np.array([[0, 0, 0, 0]] + d_o_boxes, np.float32)
    persons_np = persons_np / np.array([img_W, img_H, img_W, img_H])   
    objects_np = objects_np / np.array([img_W, img_H, img_W, img_H])
    
    return {
        'person_bbx': persons_np,
        'objects_bbx': objects_np,
        'person_bbx_score': scores_p,
        'object_bbx_score': scores_o,
        'class_id_objects': class_id_objects
    }
    
    
def get_attention_maps(segment_key, flag):
    """it gives us a map type representation for the attentions at the object level and the person level.
        we get the data for building the attention map.


    Returns:
        for all the person, object pair, we build the union box, and get the attention map for all pairs.
    """
    compact_detections = get_compact_detections(segment_key, flag)
    persons_np, objects_np = compact_detections['person_bbx'], compact_detections['objects_bbx']
    union_box = []
    no_person_dets = len(persons_np)
    no_object_dets = len(objects_np)
    for dp_i in range(no_person_dets):
        for do_i in range(no_object_dets):
            union_box.append(union_BOX(persons_np[dp_i], objects_np[do_i], segment_key))

    return np.concatenate(union_box)
        

def union_BOX(roi_pers, roi_objs, segment_key, H=64, W=64):
    
    """
    this is used for building the attention maps. it is used for getting the map co ordinates with attentions. 
    """
    assert H == W
    roi_pers = np.array(roi_pers*H, dtype=int)
    roi_objs = np.array(roi_objs*H, dtype=int) 
    sample_box = np.zeros([1, 2, H, W])
    sample_box[0, 0, roi_pers[1]:roi_pers[3]+1, roi_pers[0]:roi_pers[2]+1] = 100
    sample_box[0, 1, roi_objs[1]:roi_objs[3]+1, roi_objs[0]:roi_objs[2]+1] = 100
    
    return sample_box



def get_compact_label(segment_key, flag):
    
    """This is where the main fun starts, this compares the annotation detections, with the original object detections, by making them compared with each other, we create the robust annotations. Now, here, we,
    throughout the process of filtering, we use IoU. to get correct annotations. Instead of using bbx from annot, we use more robust bbx co-ordinates from object_det with the help of IoU relation.
    
    Returns:
       labels_np : this is of size (num_of_persons, num_of_objects+1, num_of_verbs) -> + 1 for verb without objects
       labels_single : for every possible, person object pair, we get whether there is verb or not.
    """
    
    d_p_boxes, d_o_boxes, scores_p, scores_o, class_id_persons, class_id_objects, annotation, img_shape = get_detections(segment_key, flag)
    
    no_person_dets, no_obj_dets = len(d_p_boxes), len(d_o_boxes)
    labels_np = np.zeros([no_person_dets, no_obj_dets+1, NUMBER_OF_VERBS], np.int32)
    
    a_p_boxes = [ann['person_box'] for ann in annotation]
    iou_mtx = get_iou_mtx(a_p_boxes, d_p_boxes)
    
    if no_obj_dets!=0 and len(a_p_boxes)!=0:
        max_iou_each_det = np.max(iou_mtx, axis=0)
        index_max_each_det = np.argmax(iou_mtx, axis=0)
        
        for dd in range(no_person_dets):
            cur_max_iou = max_iou_each_det[dd]
            if cur_max_iou < MATCHING_IOU:
                continue
            matched_anno = annotation[index_max_each_det[dd]]
            hoi_anns = matched_anno['hois']
            no_object_hois = [oi for oi in hoi_anns if len(oi['obj_box'])==0]
            
            for no_hoi in no_object_hois:
                verb_idx = VERB_TO_ID[no_hoi['verb']]
                labels_np[dd, 0, verb_idx] = 1
                
            object_hois = [oi for oi in hoi_anns if len(oi['obj_box'])!=0]
            a_o_boxes = [oi['obj_box'] for oi in object_hois]
            iou_mtx_o = get_iou_mtx(a_o_boxes, d_o_boxes)

            if a_o_boxes and d_o_boxes:
                for do in range(len(d_o_boxes)):
                    for ao in range(len(a_o_boxes)):
                        cur_iou = iou_mtx_o[ao, do]
                        if cur_iou < MATCHING_IOU:
                            continue
                        current_hoi = object_hois[ao]
                        verb_idx = VERB_TO_ID[current_hoi['verb']]
                        labels_np[dd, do+1, verb_idx] = 1
            
        comp_labels = labels_np.reshape(no_person_dets*(no_obj_dets+1), NUMBER_OF_VERBS)
        labels_single=np.array([1 if i.any()==True else 0 for i in comp_labels])
        labels_single=labels_single.reshape(np.shape(labels_single)[0],1)
        return{'labels_all':labels_np,'labels_single':labels_single}

    else:
        comp_labels = labels_np.reshape(no_person_dets*(no_obj_dets+1), NUMBER_OF_VERBS)
        labels_single=np.array([1 if i.any()==True else 0 for i in comp_labels])
        labels_single=labels_single.reshape(np.shape(labels_single)[0],1)
        return{'labels_all':labels_np,'labels_single':labels_single}
    
    

    

def get_iou_mtx(anns, dets):
    
    """gives us the 2 d matrix of size (n_ann, n_dets). Containing the IoU values of the bboxes
    """
    
    no_at = len(anns)
    no_dt = len(dets)
    iou_mtx = np.zeros([no_at, no_dt])
    
    for at_n in range(no_at):
        at_box = anns[at_n]
        for dt_n in range(no_dt):
            dt_box = dets[dt_n]
            iou_mtx[at_n, dt_n] = IoU_box(at_box, dt_box)
    
    return iou_mtx
            

def IoU_box(box1, box2):
    """
    Args:
        box1 : left1, top1, right1, bottom1 
        box2 : left2, top2, right2, bottom2
        
    returns:
        intersection over union
    """
    
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    left_int, top_int = max(left1, left2), max(top1, top2)
    right_int, bottom_int = min(right1, right2), min(bottom1, bottom2)
    
    area_intersection = max(0, right_int-left_int) * max(0, bottom_int-top_int)
    
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
    
    IoU = area_intersection / (area1+area2 - area_intersection)
    
    return IoU
    



def get_bad_detections(segment_key,flag):
    
    """Get detections with no persons.
    """
    
    labels_all=get_compact_label(segment_key,flag)['labels_all']
    if labels_all.size==0:
        return True
    else:
        return False
    
    
def dry_run():
	
    ALL_SEGS_train = ANNOTATIONS_TRAIN.keys()
    ALL_SEGS_val = ANNOTATIONS_VAL.keys()
    ALL_SEGS_test = ANNOTATIONS_TEST.keys()
    
    ALL_SEGS_train = [int(v) for v in ALL_SEGS_train]
    ALL_SEGS_train.sort()
    ALL_SEGS_val = [int(v) for v in ALL_SEGS_val]
    ALL_SEGS_val.sort()
    new_anns = {}
    ALL_SEGS_test = [int(v) for v in ALL_SEGS_test]
    ALL_SEGS_test.sort()
    
    bad_detections_train = []
    bad_detections_val = []
    bad_detections_test = []
    
    ######### detect bad detections #############
    
    for segkey in (ALL_SEGS_train):
        
        if get_bad_detections(segkey, "train"):
            bad_detections_train.append(segkey)
        
    
    for segkey in (ALL_SEGS_val):
        
        if get_bad_detections(segkey, "val"):
            bad_detections_val.append(segkey)
            
    
    for segkey in (ALL_SEGS_test):
        
        if get_bad_detections(segkey, 'test'):
            bad_detections_test.append(segkey)
            
    
    return bad_detections_train, bad_detections_val, bad_detections_test