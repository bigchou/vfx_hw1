import cv2, os, copy
import numpy as np
 
def MTB(dst,src,shift_bits,lower=40,upper=60):
    """Align images recursively using MTB algorithm.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]

    dst : the fixed objective image

    src : the source image

    shift_bits : search depth

    lower : ignore pixels that are between lower-th percentile and upper-th percentile

    upper : ignore pixels that are between lower-th percentile and upper-th percentile

    Returns 
    -------
    (shift_ret_xs, shift_ret_ys) : how much to move the second exposure (src) in x and y to align it with the first exposure (dst)

    Remark
    -------
    My implementation is based on the original paper:
    G. Ward, "Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures", JGT 2003
    To learn more, please refer to this link: http://www.anyhere.com/gward/papers/jgtpap2.pdf
    """
    if shift_bits == 0:
        cur_shift_x = 0
        cur_shift_y = 0
    else:
        cur_shift_x, cur_shift_y = MTB(
            cv2.resize(dst, tuple(i//2 for i in dst.shape[:2])),
            cv2.resize(src, tuple(i//2 for i in src.shape[:2])),
            shift_bits-1
        )
        cur_shift_x *= 2
        cur_shift_y *= 2
    # prepare intermediate data for dst
    dst_gray = 54.0*dst[:,:,2]+183.0*dst[:,:,1]+19.0*dst[:,:,0]#54*red + 183*green + 19*blue
    dst_lower, dst_median, dst_upper = np.percentile(dst_gray, lower), np.median(dst_gray), np.percentile(dst_gray, upper)
    dst_bin = dst_gray > dst_median
    dst_excld = ~((dst_gray > dst_lower) & (dst_upper > dst_gray))
    # prepare intermediate data for src
    src_gray = 54.0*src[:,:,2]+183.0*src[:,:,1]+19.0*src[:,:,0]#54*red + 183*green + 19*blue
    src_lower, src_median, src_upper = np.percentile(src_gray, lower), np.median(src_gray), np.percentile(src_gray, upper)
    src_bin = src_gray > src_median
    src_excld = ~((src_gray > src_lower) & (src_upper > src_gray))
    # try moving 9 directions
    h, w = src_gray.shape
    min_err = h * w
    for i in range(-1,2):
        for j in range(-1,2):
            xs = cur_shift_x + i
            ys = cur_shift_y + j
            #shift
            src_bin = src_bin.astype(np.float32)
            src_excld = src_excld.astype(np.float32)
            shifted_src_bin = cv2.warpAffine(src_bin,np.float32([[1,0,xs],[0,1,ys]]),(w,h))
            shifted_src_excld = cv2.warpAffine(src_excld,np.float32([[1,0,xs],[0,1,ys]]),(w,h))
            shifted_src_bin = (shifted_src_bin != 0)
            shifted_src_excld = (shifted_src_excld != 0)
            #xor
            diff_bin = shifted_src_bin ^ dst_bin
            #and
            diff_bin = diff_bin & shifted_src_excld
            diff_bin = diff_bin & dst_excld
            err = np.sum(diff_bin)
            if err < min_err:
                shift_ret_xs = xs
                shift_ret_ys = ys
                min_err = err
    return (shift_ret_xs, shift_ret_ys)

if __name__ == "__main__":
    # Image data courtesy of https://www.mathworks.com/help/images/ref/imregmtb.html
    depth = 6
    datafolder = os.path.join("exp","MTB")
    dstpath = "a.jpg"
    print("reading dst image: ",dstpath)
    dst = cv2.imread(os.path.join(datafolder,dstpath))
    vis = copy.deepcopy(dst)# for visualization purpose
    srcpaths = [i for i in os.listdir(datafolder) if os.path.splitext(i)[-1] == ".jpg" and i != dstpath]
    for srcpath in srcpaths:
        print("reading src image: ",srcpath)
        src = cv2.imread(os.path.join(datafolder,srcpath))
        best_x, best_y = MTB(dst, src, shift_bits=depth)
        h, w = src.shape[:2]
        align = cv2.warpAffine(src,np.float32([[1,0,best_x],[0,1,best_y]]),(w,h))
        cv2.line(align, (0, 0), (0, h), (255,0,0), 3)# draw a line for debug purpose
        vis = np.concatenate((vis,align),axis=1)
    cv2.line(vis, (0, 100), (vis.shape[1], 100), (0,255,0), 3)# draw a line for debug purpose
    outpath = "MTB_result.jpg"
    cv2.imwrite(outpath,vis)
    print("MTB result is saved to ",outpath)