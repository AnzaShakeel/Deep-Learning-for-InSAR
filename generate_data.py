import numpy as np
import h5py


def generator(train_indices, lines, itr):
        # Create empty arrays to contain batch of features and labels
        batch_size = 1
        N_IFG = 26
        batch_images = []
        clips = []
        startInd = itr*batch_size
        endInd = startInd + batch_size
        clip = np.zeros(shape=(N_IFG, 256, 256, 1))
        
        for i in range(startInd, endInd):
            startIfg = i*N_IFG
            endIfg = startIfg + N_IFG
            cnt = 0
            for j in range(startIfg, endIfg):
                folder_path = lines[np.int(train_indices[j])]
                f = h5py.File(folder_path[0:len(folder_path)-1], 'r')
                patch = np.transpose(f['patch'])     
                clip[cnt, :, :, 0] = patch
                cnt = cnt  + 1
               
            clips.append(clip)
            
        
        batch_images = np.array(clips, dtype=np.float32)
       
        return batch_images
    
    
def generator_bkwd(train_indices, lines, itr):
        # Create empty arrays to contain batch of features and labels
        batch_images = []
        batch_size = 1
        N_IFG = 26
        clips = []
        startInd = itr*batch_size
        endInd = startInd + batch_size
        clip = np.zeros(shape=(N_IFG, 256, 256, 1))
        
        for i in range(startInd, endInd):
            startIfg = i*N_IFG
            endIfg = startIfg + N_IFG
            cnt = 0
            for j in range(startIfg, endIfg):
                folder_path = lines[np.int(train_indices[j])]
                f = h5py.File(folder_path[0:len(folder_path)-1], 'r')
                patch = np.transpose(f['patch'])
                    
                clip[cnt, :, :, 0] = patch
                cnt = cnt  + 1
               
            clips.append(clip)
              
        batch_images = np.array(clips, dtype=np.float32)
        batch_images_bkwd = np.zeros(shape=(1,N_IFG, 256, 256, 1))
        
        batch_images_bkwd[0,0,:,:] = -1*batch_images[0,25,:,:]
        batch_images_bkwd[0,1,:,:] = -1*batch_images[0,24,:,:]
        batch_images_bkwd[0,2,:,:] = -1*batch_images[0,22,:,:]
        batch_images_bkwd[0,3,:,:] = -1*batch_images[0,19,:,:]
        
        batch_images_bkwd[0,4,:,:] = -1*batch_images[0,23,:,:]
        batch_images_bkwd[0,5,:,:] = -1*batch_images[0,21,:,:]
        batch_images_bkwd[0,6,:,:] = -1*batch_images[0,18,:,:]
        batch_images_bkwd[0,7,:,:] = -1*batch_images[0,15,:,:]
        
        batch_images_bkwd[0,8,:,:] = -1*batch_images[0,20,:,:]
        batch_images_bkwd[0,9,:,:] = -1*batch_images[0,17,:,:]
        batch_images_bkwd[0,10,:,:] = -1*batch_images[0,14,:,:]
        batch_images_bkwd[0,11,:,:] = -1*batch_images[0,11,:,:]
        
        batch_images_bkwd[0,12,:,:] = -1*batch_images[0,16,:,:]
        batch_images_bkwd[0,13,:,:] = -1*batch_images[0,13,:,:]
        batch_images_bkwd[0,14,:,:] = -1*batch_images[0,10,:,:]
        batch_images_bkwd[0,15,:,:] = -1*batch_images[0,7,:,:]
        
        batch_images_bkwd[0,16,:,:] = -1*batch_images[0,12,:,:]
        batch_images_bkwd[0,17,:,:] = -1*batch_images[0,9,:,:]
        batch_images_bkwd[0,18,:,:] = -1*batch_images[0,6,:,:]
        batch_images_bkwd[0,19,:,:] = -1*batch_images[0,3,:,:]
        
        batch_images_bkwd[0,20,:,:] = -1*batch_images[0,8,:,:]
        batch_images_bkwd[0,21,:,:] = -1*batch_images[0,5,:,:]
        batch_images_bkwd[0,22,:,:] = -1*batch_images[0,2,:,:]
        batch_images_bkwd[0,23,:,:] = -1*batch_images[0,4,:,:]
        batch_images_bkwd[0,24,:,:] = -1*batch_images[0,1,:,:]
        batch_images_bkwd[0,25,:,:] = -1*batch_images[0,0,:,:]
        
        batch_images_out = np.array(batch_images_bkwd, dtype=np.float32)
    
        return batch_images_out

def bkwd_names_indx(st_name, end_name):
    fwd_names = np.array(np.arange(st_name, end_name, -1))
    bkwd_names = np.array(np.zeros((fwd_names.shape)))
    bkwd_names[0] = fwd_names[0]
    bkwd_names[1] = fwd_names[1]
    bkwd_names[2] = fwd_names[3]
    bkwd_names[3] = fwd_names[6]
    bkwd_names[4] = fwd_names[2]
    bkwd_names[5] = fwd_names[4]
    bkwd_names[6] = fwd_names[7]
    bkwd_names[7] = fwd_names[10]
    bkwd_names[8] = fwd_names[5]
    bkwd_names[9] = fwd_names[8]
    bkwd_names[10] = fwd_names[11]
    bkwd_names[11] = fwd_names[14]
    bkwd_names[12] = fwd_names[9]
    bkwd_names[13] = fwd_names[12]
    bkwd_names[14] = fwd_names[15]
    bkwd_names[15] = fwd_names[18]
    bkwd_names[16] = fwd_names[13]
    bkwd_names[17] = fwd_names[16]
    bkwd_names[18] = fwd_names[19]
    bkwd_names[19] = fwd_names[22]
    
    bkwd_names[20] = fwd_names[17]
    bkwd_names[21] = fwd_names[20]
    bkwd_names[22] = fwd_names[23]
    bkwd_names[23] = fwd_names[21]
    bkwd_names[24] = fwd_names[24]
    bkwd_names[25] = fwd_names[25]
    
    return bkwd_names
