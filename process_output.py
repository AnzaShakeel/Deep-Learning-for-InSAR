import numpy as np
import matplotlib
import warnings
warnings.filterwarnings('ignore')

def save_IFG(startIfg, endIfg, lines_test, sh_indices_ts, test_set, IFG_pred, sub_folder_path):

    cnt = 0
    for ig in range(startIfg, endIfg):
            
            test_image_path = lines_test[sh_indices_ts[ig]]
            time_frame = test_image_path[len(test_image_path)-22 :len(test_image_path)-5] # extract name of interferogram
            res = test_set[0,cnt,:,:,0] - IFG_pred[0,cnt,:,:,0]

            ## save images or comparison
            filename = sub_folder_path + time_frame + '_gt.png' 
            matplotlib.image.imsave(filename, test_set[0,cnt,:,:,0]) ## GT = ground truth or input interferograms
            filename = sub_folder_path + time_frame + '_rec.png' 
            matplotlib.image.imsave(filename, IFG_pred[0,cnt,:,:,0]) ## REC = reconstructed interferograms
            filename = sub_folder_path + time_frame + '_res.png' 
            matplotlib.image.imsave(filename, res) ## RES = residual interferograms (GT- REC)
            
            cnt = cnt + 1
            
            
def get_names_TS(startIfg, endIfg, lines_test, sh_indices_ts):
    
    all_ifg = []
        
    for ig in range(startIfg, endIfg):
            
            test_image_path = lines_test[sh_indices_ts[ig]]
            
            time_frame = test_image_path[len(test_image_path)-22 :len(test_image_path)-5]  # extract name of interferogram
            all_ifg.append(np.int(time_frame[0:8]))
            all_ifg.append(np.int(time_frame[9:]))
    
    TS_names = np.unique(all_ifg)
    return TS_names


def save_EPOCHS(TS_names, TS_f, sub_folder_path1):
    for ts in range(0,9):
          
            ##### save images
            filename = sub_folder_path1 + str(TS_names[ts]) + '_rec.png' 
            #print(filename)
            matplotlib.image.imsave(filename, TS_f[0,ts,:,:,0])  ## REC = predicted epoch time-series
            
            
def save_IFG_backward(startIfg, endIfg, lines_test, sh_indices_ts, bkwd_names_in, test_set, IFG_pred, sub_folder_path):
    
    cnt = 0
    for ig in range(startIfg, endIfg, -1):
                        
            test_image_path = lines_test[sh_indices_ts[np.int(bkwd_names_in[cnt])]]
            time_frame = test_image_path[len(test_image_path)-22 :len(test_image_path)-5] # extract name of interferogram
            res = test_set[0,cnt,:,:,0] - IFG_pred[0,cnt,:,:,0]

            ##### save images
            filename = sub_folder_path + time_frame + '_gt.png'
            matplotlib.image.imsave(filename, test_set[0,cnt,:,:,0])
            filename = sub_folder_path + time_frame + '_rec.png'
            matplotlib.image.imsave(filename, IFG_pred[0,cnt,:,:,0])
            filename = sub_folder_path + time_frame + '_res.png'
            matplotlib.image.imsave(filename, res)
                 
            cnt = cnt + 1
            
            
def get_names_TS_backwards(startIfg, endIfg, lines_test, sh_indices_ts):
    
    all_ifg = []
                 
    for ig in range(startIfg, endIfg, -1):
                 
            test_image_path = lines_test[sh_indices_ts[ig]]
            time_frame = test_image_path[len(test_image_path)-22 :len(test_image_path)-5] # extract name of interferogram
            all_ifg.append(np.int(time_frame[0:8]))
            all_ifg.append(np.int(time_frame[9:]))
    TS_names = np.unique(all_ifg)
    return TS_names


def save_EPOCHS_BACKWARDS(TS_names, TS_f, sub_folder_path1):
    for ts in range(8,-1,-1):

            ##### save images
            filename = sub_folder_path1 + str(TS_names[ts]) + '_rec.png'
            #print(filename)
            matplotlib.image.imsave(filename, TS_f[0,8-ts,:,:,0])
