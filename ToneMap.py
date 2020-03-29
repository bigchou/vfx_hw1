from scipy.ndimage.filters import gaussian_filter
import cv2, copy, os
import numpy as np
from HDR import Debevec

def Reinhard2002(HDR,a=0.00125,phi=1.0,num_scale=1,saturation=1.0,isGlobalMode=True):
    """Assign colors to the pixels of the compressed dynamic range image using Reinhard 2002 approach.

    Parameters
    ----------
    HDR : the dynamic range image

    a : key (higher key, brigher intensity; lower key, darker intensity)

    phi : sharpening parameters (Local Operator Mode only)

    num_scale : number of times to perform DoF (Local Operator Mode only)

    saturation : value to controll saturation (default: 1.0)

    isGlobalMode : global_operator if True else local_operator

    Returns 
    -------
    LDR : return Low Dynamic Range image

    Remark
    -------
    My implementation is based on this paper:
    E Reinhard et al., "Photographic Tone Reproduction for Digital Images", Siggraph 2002
    To learn more, please refer to this link: http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf
    """
    def lumin2BGR(HDR,Lw,Ld,saturation=1.0):
        """Get back to BGR domain from Luminance map

        Parameters
        ----------
        HDR : the dynamic range image

        Lw : the luminance image

        Ld : the display image

        saturation : value to controll saturation (default: 1.0)

        Returns 
        -------
        LDR : return Low Dynamic Range image

        Remark
        -------
        My implementation is based on this paper:
        R. Fattal et al., "Gradient Domain High Dynamic Range Compression", Siggraph 2002
        To learn more, please refer to this link: https://www.cse.huji.ac.il/~danix/hdr/hdrc.pdf
        """
        h, w, channel = HDR.shape
        tone_map = []
        for i in range(channel):
            tone_map.append(Ld * ( (HDR[:,:,i]/Lw)**saturation ))
        tone_map = np.stack(tone_map,axis=2)
        tone_map = np.where(tone_map > 1.0, 1.0, tone_map)#clamping
        return (tone_map*255.0).astype(np.uint8)
    B, G, R = HDR[:,:,0], HDR[:,:,1], HDR[:,:,2]
    Lw=0.27*R+0.67*G+0.06*B#The calculation is based on the original paper
    h, w, channel = HDR.shape
    mean_Lw = np.exp(np.sum(np.log(Lw+1e-8))/(h*w))
    Lm = a*(Lw/mean_Lw)
    #MAIN
    Ld = None
    if isGlobalMode:
        print("Enable Global Operator Mode")
        Lwhite = np.max(Lw)#By default we set Lwhite to the maximum luminance in the scene based on the original paper
        Ld = Lm/(1.0+Lm)#equation 3
        Ld = Ld * (1.0+Lm/(Lwhite**2))#equation 4
    else:
        print("Enable Local Operator Mode")
        epsilon=0.05#the threshold set to 0.05 based on the original paper
        smax = np.zeros((h,w),dtype=np.uint8)
        mark = np.zeros((h,w),dtype=bool)
        LsBlurlist = []
        for s in range(1,num_scale+1):
            LsBlur = gaussian_filter(Lm, sigma=s)
            LsBlurlist.append(LsBlur)
            Vs = (LsBlur - gaussian_filter(Lm, sigma=s+1))/((2**phi)*a/(s**2) +  LsBlur)
            ind = np.where(Vs < epsilon)
            smax[ind] = s
        smax = np.where(smax==0,num_scale,smax)
        mark = np.zeros((h,w),dtype=bool)
        Ld = np.zeros((h,w),dtype=np.float32)
        LsBlurlist = np.array(LsBlurlist)
        for s, LsBlur in enumerate(LsBlurlist):
            ind = np.where(smax==s+1)
            tmp = np.zeros((Ld.shape),dtype=bool)
            tmp[ind] = True
            Ld = Lm / (1.0 + LsBlur*tmp)
    if Ld is None:raise Exception("Unexpected error")
    return lumin2BGR(HDR,Lw,Ld,saturation=saturation)

def Reinhard2005(HDR,f,m,a,c):
    """Assign colors to the pixels of the compressed dynamic range image using Reinhard 2005 approach.

    Parameters
    ----------
    HDR : the dynamic range image

    f : control overall intensity (the higher, the lighter)

    m : control contrast

    a : set 1 to controll pixel intensity adaptation, set 0 to controll global adapatation

    c : control amount of color correction

    Returns 
    -------
    LDR : return Low Dynamic Range image

    Remark
    -------
    My implementation is based on this paper:
    E Reinhard et al., "Dynamic range reduction inspired by photoreceptor physiology", TVCG 2005
    To learn more, please refer to this link: http://erikreinhard.com/papers/tvcg2005.pdf
    """
    h, w, channel = HDR.shape
    B, G, R = HDR[:,:,0], HDR[:,:,1], HDR[:,:,2]
    L = 0.2125*R + 0.7154*G + 0.0721*B
    retHDR = copy.deepcopy(HDR)
    # channel averages
    Cav = [np.mean(retHDR[:,:,i]) for i in range(channel)]
    Cav_ = copy.deepcopy(Cav)
    Cav = [Cav_[i] * np.ones(L.shape,dtype=np.float32) for i in range(channel)]
    Cav = np.stack(Cav,axis=2)
    # luminances
    Lav, Lmax, Lmin = np.mean(L), np.max(L), np.min(L)
    Lav_ = copy.deepcopy(Lav)
    Lav = np.stack((Lav_*np.ones(L.shape,dtype=np.float32),)*3,axis=-1)
    Llav= np.mean(np.log(L+ 1e-7))
    
    # brightness controller
    f_ = np.exp(-1.0*f)# f_ is set to 1 as an appropriate initial value in original paper (optional)
    # make m dependent on whether the image is high-key(overall light) or low-key(overall dark)
    m = m if m > 0.0 else 0.3+0.7*(((np.log(Lmax) - Llav) / (np.log(Lmax) - np.log(Lmin)))**1.4)
    
    L = np.stack((L,)*3, axis=-1)#(h,w)=>(h,w,ch)
    I_l = c * retHDR + (1-c) * L#local
    I_g = c * Cav + (1-c) * Lav#global
    I_a = a * I_l + (1-a) * I_g#pixel adaptation
    retHDR /= (retHDR + ((f_*I_a)**m))#light adaptation
    # normalize function referred to original paper
    B, G, R = retHDR[:,:,0], retHDR[:,:,1], retHDR[:,:,2]
    L = 0.2125*R + 0.7154*G + 0.0721*B
    Lmax = np.max(L)
    Lmin = np.min(L)
    if(Lmax - Lmin > 0.0):
        retHDR = np.clip( (retHDR - Lmin) / (Lmax - Lmin), 0.0 , 1.0)
    return (retHDR*255.0).astype(np.uint8)

if __name__ == "__main__":
    outpath = "result"
    datafolder = "exp"
    exp_id = "ex_2"
    annofile = os.path.join(datafolder,exp_id,"task2_ss_ex2.txt")
    if not os.path.exists(outpath):
        print("create directory ",outpath)
        os.mkdir(outpath)
    if not os.path.exists(os.path.join(outpath,exp_id)):
        print("create directory ",os.path.join(outpath,exp_id))
        os.mkdir(os.path.join(outpath,exp_id))
    # ===========================================================
    images, shutters = [], []
    with open(annofile,"r",encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:continue
            imgpath, shutter = line.split(",")
            imgpath = os.path.join(datafolder,exp_id,imgpath)
            print("reading ",imgpath)
            images.append(cv2.imread(imgpath))
            shutters.append(shutter)
    images = np.array(images)
    shutters = np.array(shutters,dtype=np.float32)
    HDR = Debevec(images,shutters,l=50,num_chosen=50,outpath=os.path.join(outpath,exp_id)).process()
    # ===========================================================
    out = os.path.join(outpath,exp_id,"Reinhard2002_Global.jpg")
    cv2.imwrite(out,Reinhard2002(HDR,a=0.6))
    print("LDR is saved to ",out)
    out = os.path.join(outpath,exp_id,"Reinhard2002_Local.jpg")
    cv2.imwrite(out,Reinhard2002(HDR,a=0.6,num_scale=10,phi=1,isGlobalMode=False))
    print("LDR is saved to ",out)
    out = os.path.join(outpath,exp_id,"Reinhard2005.jpg")
    cv2.imwrite(out,Reinhard2005(HDR,f=0.0,m=0.8,a=1.0,c=0.8))
    print("LDR is saved to ",out)
    