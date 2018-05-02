import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np

import re
import glob
import os
import json
import cv2
import tifffile
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

    def __init__(self,path=None,trench_width=17,filename_fmt=None):

        self.files=[]
        self.path=path
        self.trench_width=trench_width
        self.filename_fmt=filename_fmt
        self.ws_labels=None
        self.pos=None
        self.frame_start=0
        self.frame_limit=0
        self.is_stack=False

    
    def __enlistFiles(self,at='/',frame_start=None,frame_limit=None,pos=0,file_format='tif'):

        self.files=glob.glob(self.path+at+'*.'+file_format)
        self.files.sort()
        
        # if no frame limit specified use all frames
        if frame_start is None:
            self.frame_start=1
        if frame_limit is None:
            self.frame_limit=len(self.files)

        # decide if files stack or single
        self.is_stack=False
        self.stack=skio.imread(self.files[0])
        if len(self.stack.shape)==3:
            self.is_stack=True
            self.frame_limit=self.stack.shape[0]
            # if filename format defined locate stack file per position
            # otherwise iterate over all existing files (default)
            if self.filename_fmt is not None:
                self.files=[self.path+at+self.filename_fmt%pos]
                self.stack=skio.imread(self.files[0])
            
    def __getFrame(self,pos=0,frame=0):

        if self.is_stack:

            if self.pos!=pos:
                self.pos=pos
                self.stack=skio.imread(self.files[0])
                self.frame_limit=self.stack.shape[0]
                self.result_stack=np.zeros(self.stack.shape,dtype=np.uint8)

            return self.stack[frame-1,:,:]
        else:
            #return pl.imread(self.files[frame-1])
            return cv2.imread(self.files[frame-1],0)



    @staticmethod
    def N2spread(x):
        # get nonzero elements
        ix=np.where(x>0)[0]
        # except empty signal
        if len(ix)<1:
            return 0
        return np.mean(abs(ix[0]-ix))

    # Phase Imaging Step 1: Balancing Background Noise
    # 
    def balanceBackground(self,frame_start=None,frame_limit=None,pos=0):
    
        # create AutoCrop folder
        if not os.path.isdir(self.path+'/BackgroundBalanced'):
            os.system('mkdir "'+self.path+'/BackgroundBalanced"')

        self.__enlistFiles(at='/', frame_start=frame_start,frame_limit=frame_limit,pos=pos)

        # background balance images
        for i in tqdm(range(self.frame_start,self.frame_limit),total=self.frame_limit-self.frame_start, desc='BackgroundBalancer'):
            
            self.im=self.__getFrame(pos,i)/255.

            self.imn=np.zeros(self.im.shape)
            self.img=filters.gaussian(self.im,sigma=10)
            self.imn=(self.im-self.img)
            self.imn[self.imn<0]=0
            self.imn/=np.median(np.sort(np.ravel(self.imn))[-1000:])
            self.imn[self.imn>1]=1

            if self.is_stack:
                self.result_stack[i-1,:,:]=(self.imn*255).astype(np.uint8)
            if not self.is_stack:
                cim=Image.fromarray((self.imn*255).astype(np.uint8))
                cim.save(self.path+'/BackgroundBalanced/'+self.files[i-1].split('/')[-1])

        if self.is_stack:
            tifffile.imsave(self.path+'/BackgroundBalanced/'+self.files[0].split('/')[-1],self.result_stack)
    
    # Phase Imaging Step 2: Rotate images to get trenches straight in y-axis
    #
    def fixRotation(self,frame_start=None,frame_limit=None,pos=0):        

        def __max(x):
            if len(x)==0:
                return np.nan
            return max(x)
        def __min(x):
            if len(x)==0:
                return np.nan
            return min(x)
        def __mad(arr):
            """ Median Absolute Deviation: a "Robust" version of standard deviation.
            Indices variabililty of the sample.
            https://en.wikipedia.org/wiki/Median_absolute_deviation 
            """
            arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
            med = np.median(arr)
            return np.median(np.abs(arr - med))

        # create AutoCrop folder
        if not os.path.isdir(self.path+'/RotationFixed'):
            os.system('mkdir "'+self.path+'/RotationFixed"')

        self.__enlistFiles(at='/BackgroundBalanced/',frame_start=frame_start,frame_limit=frame_limit,pos=pos)

        for i in tqdm(range(self.frame_start,self.frame_limit),total=self.frame_limit-self.frame_start,desc='RotationFixer'):

            self.im=self.__getFrame(pos,i)/255.
            self.imn=self.im/np.max(np.max(self.im))
            
            if i==self.frame_start:
                # get a coarse mask for identfying the trench location
                im_gauss10=filters.gaussian(self.imn,5)
                coarse_mask=im_gauss10>threshold_otsu(im_gauss10)
                
                # eliminate other (occasional) particles that may exist on the frame
                skeleton=morphology.remove_small_objects(coarse_mask,min_size=self.imn.shape[0]*self.imn.shape[1]*0.02,connectivity=2)

                #pl.figure()
                #pl.imshow(skeleton)
                #pl.savefig('skeleton_pos%03d.png'%pos)
                #pl.close()

                # find upper and lower lines of the trenches
                maxline=np.array(list(map(lambda x: __max(np.where(x>0)[0]),skeleton.T)))
                maxline=maxline[~np.isnan(maxline)]
                minline=np.array(list(map(lambda x: __min(np.where(x>0)[0]),skeleton.T)))
                minline=minline[~np.isnan(minline)]
                # use only local minima to account for wells btw trenches
                peaks=argrelextrema(minline,np.less_equal,order=5)[0]
                minlinep=minline[peaks]

                # make fit to the upper trench line
                ransac = linear_model.RANSACRegressor(residual_threshold=__mad(minlinep)+1e-100)
                ransac.fit(peaks[:,np.newaxis], minlinep)
                line_X_min = np.arange(peaks.min(), peaks.max())[:, np.newaxis]
                line_y_ransac_min = ransac.predict(line_X_min)
                est_min=ransac.estimator_.coef_[0]

                #pl.figure()
                #pl.plot(minline)
                #pl.plot(peaks,minline[peaks],'x')
                #inlier_mask = ransac.inlier_mask_
                #pl.plot(peaks[inlier_mask], minlinep[inlier_mask], marker='o')
                #pl.savefig('trench_line%03d.png'%pos)
                #pl.close()


                # make fit to the lower trench line
                ransac = linear_model.RANSACRegressor(residual_threshold=__mad(maxline)+1e-100)
                ransac.fit(np.arange(1,len(maxline)+1)[:,np.newaxis], maxline)
                line_X_max = np.arange(peaks.min(), peaks.max())[:, np.newaxis]
                line_y_ransac_max = ransac.predict(line_X_max)
                est_max=ransac.estimator_.coef_[0]

                # estimate the angle difference
                est_der=(est_min)
                est_angle=180*np.arctan(est_der)/np.pi

            # rotate and save the result
            cim=Image.fromarray((self.imn*255).astype(np.uint8))
            cim=cim.rotate(est_angle,resample=Image.BICUBIC, expand=False)
            
            if self.is_stack:
                self.result_stack[i-1,:,:]=np.array(cim,dtype=np.uint8)
            if not self.is_stack:
                filename=self.files[i-1].split('/')[-1]
                cim.save(self.path+'/RotationFixed/'+filename)
                meta={'rotation_angle':str(est_angle)}
                json.dump(meta,open(self.path+'/RotationFixed/'+filename.split('.')[0]+'.meta','w'))
        
        if self.is_stack:
            tifffile.imsave(self.path+'/RotationFixed/'+self.files[0].split('/')[-1],self.result_stack)

    # Phase Imaging Step 3: Remove the stage movements using template matching
    #
    def createTemplate(self,img):
        """This function roughly locates the mother machine trenches, and
        creates a cropped template for Template Matching Algorithm."""

        # find y-axis lines
        self.spread=list((map(self.N2spread,img>threshold_otsu(img)//2)))
        self.spread=np.array(self.spread)/img.shape[1]
        self.spread=medfilt(self.spread,29)
        self.xsum=medfilt(np.sum(img,1)/img.shape[1],29)
        rough_est=np.where(self.spread>0.25)[0]
        ybot=max(rough_est)
        top_est=np.where(self.xsum[:(ybot+min(rough_est))//2]<5)[0]
        if len(top_est)>0:
            ytop=max(top_est)
        else:
            ytop=0

        # find x-axis lines
        self.xspread=list((map(self.N2spread,(img>threshold_otsu(img)//2).T)))
        self.xspread=np.array(self.xspread)/img.shape[1]
        self.xspread=medfilt(self.xspread,29)
        self.ysum=medfilt(np.sum(img,0)/img.shape[0],29)
        right_horizon=np.max(np.where(self.xspread<0.25)[0])
        left_horizon=np.min(np.where(self.xspread<0.25)[0])

        right_est=np.where(self.xspread[left_horizon:right_horizon]>0.25)[0]
        xright=left_horizon+max(right_est)
        left_est=np.where(self.xspread[left_horizon:right_horizon]>0.25)[0]
        if len(left_est)>0:
            xleft=left_horizon+min(left_est)
        else:
            xleft=0

        pl.figure()
        pl.plot(self.xspread)
        pl.plot(right_est,self.xspread[right_est],'or')
        pl.plot(xleft,self.xspread[xleft],'gx')
        pl.plot(xright,self.xspread[xright],'gx')
        pl.savefig('createtemplate_xspread_p%03d.png'%self.pos)

        pl.figure()
        pl.plot(self.ysum)
        pl.plot(left_est,self.ysum[left_est],'or')
        pl.savefig('createtemplate_ysum_p%03d.png'%self.pos)

        return img[ytop:ybot,xleft:xright]


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
    def matchTemplateBatch(self,frame_start=None,frame_limit=None,pos=0):
        
        self.__enlistFiles(at='/RotationFixed/', frame_start=frame_start,frame_limit=frame_limit,pos=pos)

        # create AutoCrop folder
        if not os.path.isdir(self.path+'/TemplateMatched'):
            os.system('mkdir "'+self.path+'/TemplateMatched"')


        # create a rough template
        template=self.createTemplate(self.__getFrame(pos=pos,frame=1))

        # read the template image
        #template=cv2.imread(self.path+'/template.tif',0)
        #self.template_g=(255*filters.gaussian(template,sigma=3)).astype('uint8')

        # crop images
        for i in tqdm(range(self.frame_start,self.frame_limit),total=self.frame_limit-self.frame_start,desc='TemplateMatching'):
    
            self.im=self.__getFrame(pos=pos,frame=i)

            #self.imc_g=(255*filters.gaussian(imc,sigma=3)).astype('uint8')
            tl,br=self.matchTemplate(self.im,template)
            
            if i==self.frame_start:
                ref_br=br
            else:
                move_x = ref_br[1]-br[1]
                move_y = ref_br[0]-br[0]
                #print(files[i-1].split('/')[-1],tl,br,move_x,move_y)
                self.im=self.moveImage(self.im,move_x,move_y)

            if self.is_stack:
                self.result_stack[i-1,:,:]=np.array(self.im,dtype=np.uint8)
            else:
                cv2.imwrite(self.path+'/TemplateMatched/'+self.files[i-1].split('/')[-1],self.im)

        if self.is_stack:
            tifffile.imsave(self.path+'/TemplateMatched/'+self.files[0].split('/')[-1],self.result_stack)

    def autoCrop(self,frame_start=None,frame_limit=None,pos=0):

        self.__enlistFiles(at='/TemplateMatched/', frame_start=frame_start,frame_limit=frame_limit,pos=pos)

        # create AutoCrop folder
        if not os.path.isdir(self.path+'/AutoCrop'):
            os.system('mkdir "'+self.path+'/AutoCrop"')

        # crop images
        for i in tqdm(range(self.frame_start,self.frame_limit),total=self.frame_limit-self.frame_start,desc='AutoCrop'):
            
            self.im=self.__getFrame(pos=pos,frame=i)
           
            if i==1:
                self.spread=list((map(self.N2spread,self.im>threshold_otsu(self.im)//2)))
                self.spread=np.array(self.spread)/self.im.shape[1]
                self.spread=medfilt(self.spread,29)
                self.xsum=medfilt(np.sum(self.im,1)/self.im.shape[1],29)
                rough_est=np.where(self.spread>0.25)[0]
                ybot=max(rough_est)

                #pl.figure()
                #pl.plot(self.spread)
                #pl.plot(rough_est,self.spread[rough_est],'xr')
                #pl.savefig('autocrop_spread_p%03d.png'%pos)
#
                #pl.figure()
                #pl.plot(self.xsum)
                #pl.savefig('autocrop_xsum_p%03d.png'%pos)

                top_est=np.where(self.xsum[:(ybot+min(rough_est))//2]<5)[0]
                if len(top_est)>0:
                    ytop=max(top_est)
                else:
                    ytop=0

                self.result_stack=np.zeros((self.stack.shape[0],ybot-ytop,self.stack.shape[2]),dtype=np.uint8)
            

            cim=Image.fromarray(self.im[ytop:ybot,:].astype(np.uint8))
            
            if self.is_stack:
                self.result_stack[i-1,:,:]=np.array(cim,dtype=np.uint8)
            if not self.is_stack:
                filename=files[i-1].split('/')[-1]
                cim.save(self.path+'/AutoCrop/'+filename)
                #meta={'ytop':str(ytop),'ybot':str(ybot)}
                #json.dump(meta,open(self.path+'/AutoCrop/'+filename.split('.')[0]+'.meta','w'))

        if self.is_stack:
            tifffile.imsave(self.path+'/AutoCrop/'+self.files[0].split('/')[-1],self.result_stack)

    @staticmethod
    def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='both', kpsh=False, valley=False, show=False, ax=None):
        """Detect peaks in data based on their amplitude and other features.

        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.
        show : bool, optional (default = False)
            if True (1), plot data in matplotlib figure.
        ax : a matplotlib.axes.Axes instance, optional (default = None).

        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.
        """

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        if show:
            if indnan.size:
                x[indnan] = np.nan
            if valley:
                x = -x
            _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

        return ind

    def kymograph(self,frame_start=None,frame_limit=None):
        
        
        if not os.path.isdir(self.path+'/Kymographs'):
            os.system('mkdir "'+self.path+'/Kymographs"')
        files=glob.glob(self.path+'/AutoCrop/*.tif')
        files.sort()

        # if no frame limit specified use all frames
        if self.frame_start is None:
            self.frame_start=1
        if self.frame_limit is None:
            self.frame_limit=len(files)

        im=pl.imread(files[self.frame_start-1]).astype(np.uint64)
        self.im=im.astype(np.uint8)
        
        peaks=[None]*self.frame_limit
        trenches=[None]*self.frame_limit
        self.logprofiles=[None]*self.frame_limit
        for i in tqdm(range(self.frame_start,self.frame_limit),total=self.frame_limit-self.frame_start,desc='TrenchProfiler'):
            im=pl.imread(files[i-1]).astype(np.uint64)
            
            ### Split Trenches
            # create log profile of timesum of the position
            self.logprofiles[i]=np.diff(medfilt(np.log10((np.sum(im,0)+1e-5)/im.shape[0]),5))
            
            peaks[i]=self.detect_peaks(self.logprofiles[i],mph=0.33,mpd=40)
            peaks[i]=np.append(peaks[i],self.detect_peaks(-self.logprofiles[i],mph=0.33,mpd=40))
            peaks[i].sort()
            
            # select trenches
            trenches[i]=np.where(np.logical_and(np.diff(peaks[i])>self.trench_width*0.9,
                        self.logprofiles[i][peaks[i][:-1]]>0))[0]
        
        # plot signal profile
        #frame=1
        #print(len(trenches[frame]))
        #fig=pl.figure(figsize=(16,5));
        #pl.plot(self.logprofiles[frame],'.-');
        #pl.plot(peaks[frame],self.logprofiles[frame][peaks[frame]],'or')
        #pl.plot(peaks[frame][trenches[frame]],self.logprofiles[frame][peaks[frame][trenches[frame]]],'og');
        ##pl.plot(peaks[frame][trenches[frame]+1],self.logprofiles[frame][peaks[frame][trenches[frame]+1]],'ob');
        #pl.xlim([0,im.shape[1]]);
        ##pl.ylim([0,2]);
        #pl.show()
        #pl.savefig('TrenchSegmentationSignal.jpg',dpi=150)
        
        # number of trenches should be same in all frames
        n_trenches=np.array([len(trenches[i]) for i in range(self.frame_start,self.frame_limit)])
        Nt=mode(n_trenches).mode[0]
        frames_to_include=np.where(n_trenches==Nt)[0]+self.frame_start
        print("%d/%d frames have slicable trenches."%(len(frames_to_include),self.frame_limit-self.frame_start+1))
        print(np.where(n_trenches!=Nt)[0]+self.frame_start)

        ### Make Trench Based Kymograph
        for tr in tqdm(range(0,Nt),total=Nt,desc='Kymographer'):
            # slice the trench, add to kymograph
            kym=[]
            meta={}
            for f in frames_to_include:
                im=skio.imread(files[f-1])
                
                # find trench coordinates
                x_start=peaks[f][trenches[f][tr]]+3
                x_end=peaks[f][trenches[f][tr]+1]-1
                trench=im[:,x_start:x_end]
                if f==self.frame_start:
                    kym=trench
                else:
                    kym=np.concatenate((kym,trench),axis=1)
                    kym=np.concatenate((kym,np.ones((kym.shape[0],1),dtype=np.uint8)*99),axis=1)
                
                meta[str(f)]={'x_start':str(x_start),'x_end':str(x_end)}
            #pl.imshow(np.invert(kym));
            
            kymi=Image.fromarray(kym,mode='L')
            kymi.save(self.path+'/Kymographs/KymographT%02d.tif'%(tr+1))
            json.dump(meta,open(self.path+'/Kymographs/KymographT%02d.meta'%(tr+1),'w'))
            
        
        return kym

    def zstack_kymographs(self):
        
        kyms=glob.glob(self.path+'/Kymographs/*.tif')
        kyms.sort()
        metas=glob.glob(self.path+'/Kymographs/*.meta')
        metas.sort()
        
        kym=[]
        # read files
        for f in range(len(kyms)):
            kym.append(pl.imread(kyms[f]))
        # find largest width
        max_width=max([k.shape[1] for k in kym])
        # create zstack
        zs=np.zeros((len(kyms),1,1,kym[0].shape[0],max_width),dtype=np.uint8)
        # put images into zstack
        for f in range(len(kyms)):
            zs[f,0,0,:,0:kym[f].shape[1]]=kym[f]

        tifffile.imsave(self.path+'/_zstack.tif', zs, compress=1, metadata={'axes': 'TCZYX'})

def win_path(path):
    wpath=path.replace("\\","\\\\")
    return wpath

def kernel(i):
    #MM=MMPhaseContrast(path="/media/sadik/PAULSSON_LAB_T3/2017_10_26_PlasmidLosses_Competition/Lane02/pos%02d"%i)
    #MM=MMPhaseContrast(path="/home/sadik/Desktop/SysBio/PAULSSON LAB/Luis/Ti3/180301_MMCompetitionSB5-SB8_NowWithDM25/Growth/line_01_s_%03d"%i)
    #MM=MMPhaseContrast(path="/Volumes/PAULSSON_LAB_T3/VIBRIO_SAMPLER_H2_05--37C--1_9/Lane_01_40m_CROP",filename_fmt="Lane_01_pos_%03d_40m_BF.tif")
    #MM=MMPhaseContrast(path="/mnt/f/VIBRIO_SAMPLER_H2_05--37C--1_9/Lane_01_40m_CROP",filename_fmt="Lane_01_pos_%03d_40m_BF.tif")
    print('pos%03d: Initialized.'%i)
    #MM.balanceBackground(pos=i)
    #MM.fixRotation(pos=i)
    MM.matchTemplateBatch(pos=i)
    #MM.autoCrop(pos=i)
    #MM.kymograph()
    #MM.zstack_kymographs()
    return i

def main():

    # run for each position
    position_start=1
    position_end=19
    # number of parallel cores to divide the work
    n_cores=1
    
    kernel(16)
    # run the multiprocess pool
    # with Pool(n_cores) as p:
    
    
    #     for i in tqdm(p.imap_unordered(kernel,list(range(position_start,position_end+1))),desc='Position'):
    #         print('Position %3d OK.'%i)
    #         pass
            

if __name__ == "__main__":
    main()