import cv2, os, random, matplotlib
import numpy as np
import matplotlib.pyplot as plt

class Debevec:
    """HDR imaging using Debevec algorithm.

    Parameters
    ----------
    HDR : the dynamic range image

    images : list of images with shape (batch, height, weight, channel)

    shutters : list of shutter speeds

    l : set larger l to emphasize the smoothness term (the response curve will becomre more smooth)

    min_chosen : number of chosen pixel location should at least more than min_chosen

    outpath : the path to the folder used for saving response curve image

    Remark
    -------
    My implementation is based on this paper:
    Paul E. Debevec et al., "Recovering High Dynamic Range Radiance Maps from Photographs", Siggraph 1997
    To learn more, please refer to this link: http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
    """
    def __init__(self,images,shutters,l,min_chosen,outpath):
        # Define variables
        self.B = np.log(shutters)#ln(deltaT_j)
        self.l = l#interpolation weight
        self.w = self.w()#weight function
        self.min_chosen = min_chosen#minimum of number of chosen pixel location
        self.outpath = outpath#output path for response curve
        self.images = images#list of images

    def process(self,color=["Blue","Green","Red"],drawCurve=False):
        # MAIN
        HDR = []
        channel = self.images.shape[-1]
        for i in range(channel):
            print("handle ",color[i]," channel.......................................")
            Z = self.pick(self.images[:,:,:,i],min_chosen=self.min_chosen)#pick pixel location
            g = self.getG(Z,self.B,self.l)
            if drawCurve:self.drawCurve(Z,g,i)
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

    def pick(self,images,min_chosen=50):
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
        if len(Z) < min_chosen:
            raise Exception("[ERROR] number of chosen locations should be more than %d"%(min_chosen))
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

def drawHDR(HDR,outfolder,color=["Blue","Green","Red"]):
    from mpl_toolkits import axes_grid1
    def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
        """Add a vertical color bar to an image plot.
        
        Remark
        -------
        This function is referred to the Matthias's implementation
        To learn more, please see this link: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        """
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        return im.axes.figure.colorbar(im, cax=cax, **kwargs)
    for i in range(HDR.shape[-1]):
        im = plt.imshow(HDR[:,:,i],cmap="jet",norm=matplotlib.colors.PowerNorm(gamma=0.14))
        add_colorbar(im)#plt.colorbar()
        out = os.path.join(outfolder,color[i]+".png")
        plt.title(color[i]+"_channel")
        plt.savefig(out)
        plt.close()
        print("save HDR to ",out)

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
    HDR = Debevec(images,shutters,l=50,min_chosen=50,outpath=os.path.join(outpath,exp_id)).process(drawCurve=True)
    drawHDR(HDR,os.path.join(outpath,exp_id))