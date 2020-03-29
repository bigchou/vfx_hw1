import cv2, os, random
import numpy as np
import matplotlib.pyplot as plt

class Debevec:
    """HDR imaging using Debevec algorithm.

    """
    def __init__(self,images,shutters,l,num_chosen,outpath):
        # Define variables
        self.B = np.log(shutters)#ln(deltaT_j)
        self.l = l#interpolation weight
        self.w = self.w()#weight function
        self.num_chosen = num_chosen#number of chosen pixel location
        self.outpath = outpath#output path for response curve
        self.images = images#list of images

    def process(self,color=["Blue","Green","Red"]):
        # MAIN
        HDR = []
        channel = self.images.shape[-1]
        for i in range(channel):
            print("handle ",color[i]," channel.......................................")
            Z = self.pick(self.images[:,:,:,i],num_chosen=self.num_chosen)#pick pixel location
            g = self.getG(Z,self.B,self.l)
            self.drawCurve(Z,g,i)
            radimap = self.getRadiMap(self.images[:,:,:,i],g)
            HDR.append(radimap)
        HDR = np.stack(HDR,axis=2)
        return HDR.astype(np.float32)
    
    def w(self):
        w = []
        for z in range(256):
            if z <= 127:
                w.append(1e-4 if z == 0 else float(z)/127.0)
            else:
                w.append(1e-4 if z == 255 else float(255.0-z)/127.0)
        return np.array(w)

    def pick(self,images,num_chosen=50):
        img = images[len(images)//2]#middle one should be more plausible
        # Find edge area
        edge = cv2.Canny(img,100,200)
        dilate = ~cv2.dilate(edge,np.ones((5,5),np.uint8),iterations=10)
        # Build position table
        pos_table = [[] for i in range(256)]
        numimg, h, w = images.shape
        row_start, row_end = int(h*0.1), int(h*0.9)#exclude padding areas introduced from MTB
        col_start, col_end = int(w*0.1), int(w*0.9)#exclude padding areas introduced from MTB
        for i in range(row_start, row_end):
            for j in range(col_start,col_end):
                if dilate[i,j]:
                    pos_table[img[i,j]].append((i,j))
        # Build Z which contains the pixel values of pixel location number i in image j
        Z = []
        for i in range(256):
            if not pos_table[i]:continue
            row, col = random.choice(pos_table[i])
            Z.append([images[i,row,col] for i in range(numimg)])
        if len(Z) < num_chosen:
            raise Exception("[ERROR] number of chosen locations should be more than %d"%(num_chosen))
        print("select %d locations"%(len(Z)))
        return np.array(Z,dtype=np.uint8)

    def drawCurve(self,Z,g,i,color_bar=["bo","go","ro"]):
        plt.figure()
        pixel_value_z = np.array([Zij for Zi in Z for Zij in Zi])
        exposure = g[pixel_value_z]
        plt.plot(exposure, pixel_value_z, color_bar[i])
        plt.xlabel('log exposure X')
        plt.ylabel('pixel value Z')
        out = os.path.join(self.outpath,"ResponseCurve_%s.png"%(color_bar[i]))
        print("save response curve to ",out)
        plt.savefig(out)
        plt.close()
    
    def getG(self,Z,B,l):
        n = 256
        p = Z.shape[0]#the point you pick. The variable p is corresponding to "n" in the slide
        m = Z.shape[1]#number of images. The variable m corresponding to "p" in the slide
        A = np.zeros((m*p+n-1, n+p), dtype=np.float32)
        b = np.zeros((A.shape[0],1),dtype=np.float32)
        # Include the data-fitting equations
        k = 0
        for i in range(p):
            for j in range(m):
                wij = self.w[Z[i,j]]
                A[k,Z[i,j]] = 1 * wij
                A[k,n+i] = -1 * wij
                b[k,0] = B[j] * wij#B[j] = ln(deltaT_j)
                k+=1
        # Fix the curve by setting its middle value to 0
        A[k, 127] = 1
        k+=1
        # Include the smoothness equations
        for i in range(n-2):
            wi = self.w[i+1]#No need to calc. 2nd differentiation on 0 and 255
            A[k,i] = 1 * l * wi
            A[k,i+1] = -2 * l * wi
            A[k,i+2] = 1 * l * wi
            k+=1
        # Solve the system using SVD
        x = np.linalg.lstsq(A,b,rcond=None)[0]
        g = x[0:n]
        return g[:,0]
    
    def getRadiMap(self,images,g):
        wZij = self.w[images]
        sum_wZij = np.sum(wZij,axis=0)+1e-8
        radimap = None
        for j in range(len(images)):
            lnE = (wZij[j] * (g[images[j]] - self.B[j])) / sum_wZij
            radimap = (lnE if j == 0 else radimap + lnE)
            if radimap is None: raise Excption("[ERROR] Radiance Map is None")
        return np.exp(radimap)

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
    cv2.imwrite(
        os.path.join(outpath,exp_id,"HDR.hdr"),
        cv2.normalize(HDR, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    )#see results just applying minmax normalization