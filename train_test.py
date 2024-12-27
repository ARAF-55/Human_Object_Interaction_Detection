import torch
import torch.nn as nn 
import time 
import errno 
import os 
import gc 
import pickle 
import shutil 
import json 
import pandas as pd 
from skimage import io, transform 
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import healper_process as helper 
import prior_vcoco as prior
import pred_vis as viss
import proper_inference as prop
import calculate_ap_classwise as ap
from tqdm import tqdm 




sigmoid = nn.Sigmoid()


### Fixing seeds ####

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

seed = 10
torch.manual_seed(seed)


np.random.seed(seed)
random.seed(seed)

softmax = nn.Softmax()

##########################

###################  parameters for person to object class mapping #############################

SCORE_TH = 0.6
SCORE_OBJ = 0.3
epoch_to_change = 400
thres_hold = -1

###############################################################################################

############# Loss function defination ########################################################

loss_com = nn.BCEWithLogitsLoss(reduction='sum')
loss_com_class = nn.BCEWithLogitsLoss(reduction='none')
loss_com_combine = nn.BCELoss(reduction='none')
loss_com_single = nn.BCEWithLogitsLoss(reduction='sum')

###############################################################################################

no_of_classes = 29

##### Helper function ######

#### Fixing the seeds for all threads ###########

def _init_fn(worker_id):
    np.random.seed(int(seed))
    
    
################################################

############# Extending number of people #####################################################

def extend(inputt, extend_number):
    
    res = np.zeros([1, np.shape(inputt)[-1]])
    
    for a in inputt:
        x = np.repeat(a.reshape(1, np.shape(inputt)[-1]), extend_number, axis=0)
        res = np.concatenate([res, x], axis=0)
        
    return res[1:]


def extend_object(inputt, extend_number):
    
    res = np.zeros([1, np.shape(inputt)[-1]])
    
    x = np.array(inputt.tolist()*extend_number)
    res = np.concatenate([res, x], axis=0)
    
    return res[1:]


################################# Filtering the results for preparing the output as per V-COCO ######################

def filtering(predicted_HOI, true, persons_np, objects_np, filters, pairs_info, image_id):
    
    res1 = np.zeros([1, no_of_classes])
    res2 = np.zeros([1, no_of_classes])
    res3 = np.zeros([1, no_of_classes])
    res4 = np.zeros([1, 4])
    res5 = np.zeros([1, 4])
    dict_1 = {}
    a = 0
    
    increment = [int(i[0] * i[1]) for i in pairs_info]
    start = 0
    
    for index, i in enumerate(filters):
        
        res1 = np.concatenate([res1, predicted_HOI[index].reshape(1, no_of_classes)], axis=0)
        res2 = np.concatenate([res2, true[index].reshape(1, no_of_classes)], axis=0)
        res3 = np.concatenate([res3, predicted_HOI[index].reshape(1, no_of_classes)], axis=0)
        res4 = np.concatenate([res4, persons_np[index].reshape(1, 4)], axis=0)
        res5 = np.concatenate([res5, objects_np[index].reshape(1, 4)], axis=0)
        
        if index == start + increment[a] - 1:
            
            dict_1[int(image_id[a]), 'score'] = res3[1:]
            dict_1[int(image_id[a]), 'pers_bbx'] = res4[1:]
            dict_1[int(image_id[a]), 'obj_bbx'] = res5[1:]
            res3 = np.zeros([1, no_of_classes])
            res4 = np.zeros([1, 4])
            res5 = np.zeros([1, 4])
            start += increment[a]
            a += 1
            
    return dict_1

########################################################################################################################

########### Saving Checkpoint ##########################

def save_checkpoint(state, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)
    
######################################################

########## LIS function from https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network #####################

def LIS(x, T, k, w):
    return T/(1+np.exp(k-w*x))

########################################################################################################################

def train_test(model, optimizer, scheduler, dataloader, number_of_epochs, break_point, saving_epoch, folder_name, batch_size, infr, start_epoch, mean_best, visualize):
    
    ################### Creating the folder where the result will be stored ######################################
    try:
        os.mkdir(folder_name)
    
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    file_name = folder_name + '/' + 'result.pickle'
    
    ###################################################################################################################
    
    loss_epoch_train = []
    loss_epoch_val = []
    loss_epoch_test = []
    initial_time = time.time()
    result = []
    
    ############## Freeing out the cache memories from cpu and gpus and declaring the phases ##########################
    
    torch.cuda.empty_cache()
    phases = ['train', 'val', 'test']
    
    if infr == 't' and visualize == 'f': ###### If running from a pretrained model for saving the best result #########
        start_epoch = start_epoch - 1
        phases = ['test']
        end_of_epochs = start_epoch + 1
        print("Only doing testing for storing results from a model")
        
    elif visualize != 'f':
        if visualize not in phases:
            print("Error! Asked to show visualization from a unknown set. The choice should be among train, val, test")
            return
    
        else:
            phases = [visualize]
            end_of_epochs = start_epoch + 1
            print("only showing predictions from a model")
            
    else:
        end_of_epochs = start_epoch + number_of_epochs
        
    ###################################################################################################################
    
    ############################ Starting the epochs #################################################################
    
    for epoch in range(start_epoch, end_of_epochs):
        
        scheduler.step() 
        print('Epoch {}/{}'.format(epoch+1,end_of_epochs))
        print('-' * 10)
        initial_time_epoch = time.time()
        
        for phase in phases:
            
            if phase == 'train':
                model.train()
            
            elif phase == 'val':
                model.train()
            
            else:
                model.eval()
                

            print('In {}'.format(phase))
            
            detections_train = []
            detections_val = []
            detections_test = []
            
            true_scores_class = np.ones([1, 80], dtype=int)
            true_scores = np.ones([1, 29], dtype=int)
            true_scores_single = np.ones([1, 1], dtype=int)
            predicted_scores = np.ones([1, 29], dtype=float)
            predicted_scores_single = np.ones([1, 1], dtype=float)
            predicted_scores_class = np.ones([1, 80], dtype=float)
            acc_epoch = 0
            iteration = 1
            
            torch.cuda.empty_cache()

            ########### Starting the iterations ###################################################################
            
            for iter, i in enumerate(tqdm(dataloader[phase])):
                
                if iter % 20 == 0:
                    torch.cuda.empty_cache()
                    
                inputs = i[0].to(device)
                labels = i[1].to(device)
                labels_single = i[2].to(device)
                image_id = i[3]
                pairs_info = i[4]
                min_batch_size = len(pairs_info)
                
                optimizer.zero_grad()
                
                if phase == 'train':
                    nav = torch.tensor([[0, epoch]]*min_batch_size).to(device)
                    
                elif phase == 'val':
                    nav = torch.tensor([[1, epoch]]*min_batch_size).to(device)
                
                else:
                    nav = torch.tensor([[2, epoch]]*min_batch_size).to(device)
                    
                true = (labels.data).cpu().numpy()
                true_single = (labels_single.data).cpu().numpy()
                
                
                
                with torch.set_grad_enabled(phase=='train' or phase=='val'):
                    
                    model_out = model(inputs, pairs_info, pairs_info, image_id, nav, phase)
                    outputs = model_out[0]         ## Visual
                    outputs_single = model_out[1]
                    outputs_combine = model_out[2] ## graph
                    outputs_gem = model_out[3]     ## attention
                    
                    predicted_HOI = sigmoid(outputs).data.cpu().numpy()
                    predicted_HOI_combine = sigmoid(outputs_combine).data.cpu().numpy()
                    predicted_single = sigmoid(outputs_single).data.cpu().numpy()
                    predicted_gem = sigmoid(outputs_gem).data.cpu().numpy()
                    predicted_HOI_pair = predicted_HOI
                    
                    
                    start_index = 0
                    start_obj = 0
                    start_pers = 0
                    start_tot = 0
                    pers_index = 1
                    persons_score_extended = np.zeros([1, 1])
                    objects_score_extended = np.zeros([1, 1])
                    class_ids_extended = np.zeros([1, 1])
                    persons_np_extended = np.zeros([1, 4])
                    objects_np_extended = np.zeros([1, 4])
                    start_no_obj = 0
                    class_ids_total = []
                    
                    
                    ################### Extending persons, obj scores to multiply with all pairs ######################
                    
                    for batch in range(len(pairs_info)):
                        
                        persons_score = []
                        objects_score = []
                        class_ids = []
                        
                        this_image = int(image_id[batch])
                        
                        scores_total = helper.get_compact_detections(this_image, phase)
                        persons_score, objects_score, persons_np, objects_np, class_ids = scores_total['person_bbx_score'], scores_total['object_bbx_score'], scores_total['person_bbx'], scores_total['objects_bbx'], scores_total['class_id_objects']
                        
                        objects_score.insert(0, float(1))
                        
                    
                        temp_scores = extend(np.array(persons_score).reshape(len(persons_score), 1), int(pairs_info[batch][1]))
                        persons_score_extended = np.concatenate([persons_score_extended, temp_scores])
                        
                        temp_scores = extend(persons_np, int(pairs_info[batch][1]))
                        persons_np_extended = np.concatenate([persons_np_extended, temp_scores])
                        
                        temp_scores = extend_object(np.array(objects_score).reshape(len(objects_score), 1), int(pairs_info[batch][0]))
                        objects_score_extended = np.concatenate([objects_score_extended, temp_scores])
                        
                        temp_scores = extend_object(objects_np, int(pairs_info[batch][0]))
                        objects_np_extended = np.concatenate([objects_np_extended, temp_scores])
                        
                        temp_scores = extend_object(np.array(class_ids).reshape(len(class_ids),1),int(pairs_info[batch][0]))
                        class_ids_extended = np.concatenate([class_ids_extended, temp_scores])
                        class_ids_total.append(class_ids)
                        
                        
                        
                        start_pers += int(pairs_info[batch][0])
                        start_obj += int(pairs_info[batch][1])
                        start_tot = start_tot + int(pairs_info[batch][1]) * int(pairs_info[batch][0])
                        
                    
                    ####################################################################################################
                    
                    ########## Applying LIS ##################################
                    
                    persons_score_extended = LIS(persons_score_extended, 8.3, 12, 10)
                    objects_score_extended = LIS(objects_score_extended, 8.3, 12, 10)  
                    
                    ###########################################################
                    
                    ########## Multiplying the scores from different streams along with the prior function from ican ###  
                    
                    
                    
                    predicted_HOI = (predicted_HOI*predicted_HOI_combine*predicted_single*predicted_gem*persons_score_extended[1:]*objects_score_extended[1:])
                    
                    loss_mask = prior.apply_prior(class_ids_extended[1:], predicted_HOI)
                    predicted_HOI = loss_mask * predicted_HOI
                    
                    ######## Calculating loss ###################################################################
                    
                    N_b = min_batch_size * 29
                    hum_obj_mask = torch.Tensor(objects_score_extended[1:]*persons_score_extended[1:]*loss_mask).to(device)
                    
                    lossf = torch.sum(loss_com_combine(sigmoid(outputs)*sigmoid(outputs_combine)*sigmoid(outputs_single)*hum_obj_mask*sigmoid(outputs_gem), labels.float())) / N_b
                    
                    lossc = lossf.item()
                    
                    acc_epoch += lossc 
                    iteration += 1
                    
                    if phase == 'train' or phase == 'val':
                        lossf.backward()
                        optimizer.step()                 
                    
                    #################################################################################################
                    
                    del lossf 
                    del model_out 
                    del inputs
                    del outputs
                    del labels
    
                ########################### If visualization ######################################################
                
                if visualize != 'f':
                    viss.visual(image_id, phase, pairs_info, predicted_HOI, predicted_single, objects_score_extended[1:], persons_score_extended[1:], predicted_HOI_combine, predicted_HOI_pair, true)
                    
                ###################################################################################################
                
                
                ########## preparing for storing results #########################################################
                predicted_scores = np.concatenate((predicted_scores, predicted_HOI), axis=0)
                true_scores = np.concatenate((true_scores, true), axis=0)
                predicted_scores_single = np.concatenate((predicted_scores_single, predicted_single), axis=0)
                true_scores_single = np.concatenate((true_scores_single, true_single), axis=0)
                ################################################################################################
                
                ################### Storing the result in V-COCO format ########################################
                
                if phase == 'test':
                    
                    if (epoch+1)%saving_epoch==0 or infr=='t':
                        
                        all_scores = filtering(predicted_HOI, true, persons_np_extended[1:], objects_np_extended[1:],predicted_single, pairs_info, image_id)
                        
                        prop.infer_format(image_id, all_scores, phase, detections_test, pairs_info)
                        
                ###############################################################################################
                
                
                
                ################# Breaking in particular number of epoch #####################################
                
                if iteration == break_point + 1:
                    
                    break
                
                ##############################################################################################
                
            if phase == 'train':
                
                loss_epoch_train.append((acc_epoch))
                
                AP, AP_single = ap.class_AP(predicted_scores[1:,:],true_scores[1:,:],predicted_scores_single[1:,],true_scores_single[1:,])
                
                AP_train = pd.DataFrame(AP,columns =['Name_TRAIN', 'Score_TRAIN'])
                AP_train_single = pd.DataFrame(AP_single,columns =['Name_TRAIN', 'Score_TRAIN'])
                
            elif phase == 'val':
                
                loss_epoch_val.append((acc_epoch))
                AP,AP_single=ap.class_AP(predicted_scores[1:,:],true_scores[1:,:],predicted_scores_single[1:,],true_scores_single[1:,])
                
                AP_val = pd.DataFrame(AP,columns =['Name_VAL', 'Score_VAL'])
                AP_val_single = pd.DataFrame(AP_single,columns =['Name_VAL', 'Score_VAL'])
                
            elif phase == 'test':
                
                loss_epoch_test.append((acc_epoch))
                AP,AP_single=ap.class_AP(predicted_scores[1:,:],true_scores[1:,:],predicted_scores_single[1:,],true_scores_single[1:,])
                AP_test = pd.DataFrame(AP,columns =['Name_TEST', 'Score_TEST'])
                AP_test_single = pd.DataFrame(AP_single,columns =['Name_TEST', 'Score_TEST'])
                
                if (epoch+1)%saving_epoch==0 or infr=='t':
                    file_name_p = folder_name+'/'+'test{}.pickle'.format(epoch+1)
                    with open(file_name_p, 'wb') as handle:
                        pickle.dump(detections_test, handle)
                
                
        
        ##################################### Saving the model ########################################################
        
        mean=AP_test.to_records(index=False)[29][1]
        
        ##### Best Model ############
        
        if mean>mean_best and infr!='t':
            
            mean_best = mean
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_best': mean_best,
                'optimizer': optimizer.state_dict(),
                'scheduler':scheduler.state_dict()
            }, filename=folder_name+'/'+'bestcheckpoint.pth.tar')
            
        ############################
        
        if (epoch+1)%saving_epoch==0  and infr!='t':
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_best': mean_best,
                'optimizer': optimizer.state_dict(),
                'scheduler':scheduler.state_dict()
            }, filename = folder_name+'/'+str(epoch + 1)+'_checkpoint.pth.tar')
            
        ###########################
        
        if infr=='t':
            AP_final=pd.concat([AP_test],axis=1)
            AP_final_single=pd.concat([AP_test_single],axis=1)
            result.append(AP_final)
            with open(file_name, 'wb') as handle:
                pickle.dump(result, handle)
                
        else:
            AP_final=pd.concat([AP_train,AP_val,AP_test],axis=1)
            AP_final_single=pd.concat([AP_train_single,AP_val_single,AP_test_single],axis=1)
            result.append(AP_final)
            with open(file_name, 'wb') as handle:
                pickle.dump(result, handle)
             
        
        time_elapsed = time.time() - initial_time_epoch
        print('APs in EPOCH:{}'.format(epoch+1))
        print(AP_final)	
        print(AP_final_single)
        
        try:
            print('Loss_train:{},Loss_validation:{},Loss_test:{}'.format(loss_epoch_train[epoch-start_epoch],loss_epoch_val[epoch-start_epoch],loss_epoch_test[epoch-start_epoch]))
        
        except:
            print('Loss_test:{}'.format(loss_epoch_test[epoch-start_epoch]))
        
        print('This epoch completes in {:.0f}m {:.06f}s'.format(
		      	        time_elapsed // 60, time_elapsed % 60))
        
        if infr=='t':
            break   
        
    
    time_elapsed = time.time() - initial_time
    print('The whole process runs for {:.0f}h {:.0f}m {:0f}s'.format(time_elapsed //3600, (time_elapsed % 3600) // 60,((time_elapsed % 3600)%60)%60)) 
    
    return    
             

        
        