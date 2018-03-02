import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np

import re
import glob
import os
import json
from PIL import Image
from tqdm import tqdm
from skimage import filters
from skimage import exposure
from skimage import morphology
from skimage import io as skio
from skimage.segmentation import active_contour
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from sklearn import linear_model
from scipy.signal import medfilt,argrelextrema
from scipy.misc import toimage
from scipy.stats import mode
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from multiprocessing import Pool

class MMPhaseContrast():

    def __init__(self,path=None,trench_width=17):

        self.path=path
        self.trench_width=trench_width
        self.ws_labels=None
        
    def matchTemplate(self,img,template):
        """
        Takes an image and a template to search for and returns bottom right
        and top left coordinates. (top_left,bottom_right) ((int,int),(int,int))
        """
        w,h=template.shape[::-1]

        #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        method=cv2.TM_CCOEFF

        # Apply template Matching
        self.res=cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left=min_loc
        else:
            top_left=max_loc

        bottom_right=(top_left[0]+w,top_left[1]+h)

        return (top_left,bottom_right)
    
    @staticmethod
    def moveImage(im,move_x,move_y,pad=0):
        """
        Moves the image without changing frame dimensions, and
        pads the edges with given value (default=0).
        """
        
        if move_y>0:
            ybound=-move_y
        else:
            ybound=im.shape[1]
        
        if move_x>0:
            xbound=-move_x
        else:
            xbound=im.shape[0]
                
        if move_x>=0 and move_y>=0:
            im[move_x:,move_y:]=im[:xbound,:ybound]
            im[0:move_x,:]=pad
            im[:,0:move_y]=pad
        if move_x<0 and move_y>=0:
            im[:move_x,move_y:]=im[-move_x:,:ybound]
            im[move_x:,:]=pad
            im[:,0:move_y]=pad
        if move_x>=0 and move_y<0:
            im[move_x:,:move_y]=im[:xbound,-move_y:]
            im[0:move_x,:]=pad
            im[:,move_y:]=pad
        if move_x<0 and move_y<0:
            im[:move_x,:move_y]=im[-move_x:,-move_y:]
            im[move_x:,:]=pad
            im[:,move_y:]=pad
        
        return im
    
    def matchTemplateBatch(self,frame_start=None,frame_limit=None):
        
        files=glob.glob(self.path+'/*.tif')
        files.sort()
        
        # if no frame limit specified use all frames
        if frame_start is None:
            frame_start=1
        if frame_limit is None:
            frame_limit=len(files)+1

        # create AutoCrop folder
        if not os.path.isdir(self.path+'/TemplateMatched'):
            os.system('mkdir "'+self.path+'/TemplateMatched"')
            
        # read the template image
        template=cv2.imread(self.path+'/template.tif',0)
        #self.template_g=(255*filters.gaussian(template,sigma=3)).astype('uint8')

        # crop images
        for i in tqdm(range(frame_start,frame_limit),total=frame_limit-frame_start,desc='TemplateMatching'):
    
            im=pl.imread(files[i-1])
            
            imc=cv2.imread(files[i-1],0)
            #self.imc_g=(255*filters.gaussian(imc,sigma=3)).astype('uint8')
            tl,br=self.matchTemplate(imc,template)
            
            if i==frame_start:
                ref_br=br
            else:
                move_x = ref_br[1]-br[1]
                move_y = ref_br[0]-br[0]
                #print(files[i-1].split('/')[-1],tl,br,move_x,move_y)
                imc=self.moveImage(imc,move_x,move_y)
            cv2.imwrite(self.path+'/TemplateMatched/'+files[i-1].split('/')[-1],imc)


def win_path(path):
    wpath=path.replace("\\","\\\\")
    return wpath

def kernel(i):
    MM=MMPhaseContrast(path="/media/sadik/PAULSSON_LAB_T3/2017_10_26_PlasmidLosses_Competition/Lane02/pos%02d"%i)
    MM.matchTemplateBatch(frame_start=1,frame_limit=None)

def main():
    # run for each position
    with Pool(2) as p:

        for i in tqdm(p.imap_unordered(kernel,list(range(1,3))),desc='Position'):
            pass
            

if __name__ == "__main__":
    main()