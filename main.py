import cv2, os, argparse, sys, time, json
import numpy as np
from HDR import Debevec
from ToneMap import Reinhard2002, Reinhard2005
from MTB import MTB

def loadData(annofile,imgfolder):
    images, shutters = [], []
    with open(annofile,"r",encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:continue
            imgpath, shutter = line.split(",")
            imgpath = os.path.join(imgfolder,imgpath)
            print("reading ",imgpath)
            images.append(cv2.imread(imgpath))
            shutters.append(shutter)
    return np.array(images), np.array(shutters,dtype=np.float32)

parser = argparse.ArgumentParser(description='VFX HW1')
parser.add_argument("--annofile",type=str,help="path to the file containing image path and shutter speed")
parser.add_argument("--imgfolder",type=str,help="path to the folder containing list of images with different shutter speeds")
parser.add_argument("--outfolder",type=str,help="path to the folder storing your results")
# MTB-specific arguments
parser.add_argument('--MTB',action="store_true")
parser.add_argument("--depth",default=6,type=int,help="search depth used in MTB")
# Devebec-specific arguments
parser.add_argument("--l",default=50,type=int,help="larger value means the response curve is more smooth")
parser.add_argument("--min_chosen",default=50,type=int,help="number of chosen pixel location should at least more than min_chosen")
# ToneMap-specific arguments
parser.add_argument("--tone_method",default="Reinhard2002Global",help="the method should be Reinhard2002Global, Reinhard2002Local or Reinhard2005 only")
parser.add_argument("--key",default="0.5",type=float,help="refer to alpha term in Reinhard2002 or f term in Reinhard2005 (the higher, the lighter)")
parser.add_argument("--saturation",default="1.0",type=float,help="saturation term (only effect Reinhard2002)")
# Reinhard2002Local-specific arguments
parser.add_argument("--phi",default="1.0",type=float,help="control sharpness in Reinhard2002 Local Operator")
parser.add_argument("--num_scale",default="1",type=int,help="number of times to perform DoG in Reinhard2002 Local Operator")
# Reinhard2005-specific arguments
parser.add_argument("--contrast",default="0.6",type=float,help="control contrast term(i.e. m) in the Reinhard2005")
parser.add_argument("--adaptation",default="1.0",type=float,help="closing to 1 means pixel intensity adaptation or closing to 0 to means global adapatation")
parser.add_argument("--chromatic",default="0.8",type=float,help="control amount of color correction")

if __name__ == "__main__":
    arg = parser.parse_args()
    exp_time = int(time.time())
    if not os.path.exists(arg.outfolder):os.mkdir(arg.outfolder)
    images, shutters = loadData(annofile=arg.annofile,imgfolder=arg.imgfolder)
    # alignment stage
    if arg.MTB:
        mid_ind = len(images)//2
        mid = images[mid_ind]
        for i,img in enumerate(images):
            if i == mid_ind:continue
            best_x, best_y = MTB(mid, img, shift_bits=depth)
            h, w = img.shape[:2]
            align = cv2.warpAffine(img,np.float32([[1,0,best_x],[0,1,best_y]]),(w,h))
            images[i] = align
    # HDR stage
    HDR = Debevec(images,shutters,l=arg.l,min_chosen=arg.min_chosen,outpath=arg.outfolder).process()
    # Tone Mapping stage
    out = os.path.join(arg.outfolder,arg.tone_method+".jpg")
    if arg.tone_method == "Reinhard2002Global":
        cv2.imwrite(out,Reinhard2002(HDR,a=arg.key))
    elif arg.tone_method == "Reinhard2002Local":
        cv2.imwrite(out,Reinhard2002(HDR,a=arg.key,num_scale=arg.num_scale,phi=arg.phi,isGlobalMode=False))
    elif arg.tone_method == "Reinhard2005":
        cv2.imwrite(out,Reinhard2005(HDR,f=arg.key,m=arg.contrast,a=arg.adaptation,c=arg.chromatic))
    else:
        raise Exception("[ERROR] the tone_method should be Reinhard2002Global, Reinhard2002Local or Reinhard2005 only")
    print("the result is saved to ",out)
    with open(os.path.join(arg.outfolder,"info_%d.txt"%(exp_time)),"w",encoding="utf-8") as f:
        f.write('\n'.join(sys.argv[1:]))
        f.write('\n=============================\n')
        json.dump(arg.__dict__, f, indent=2)
    print("DONE")

