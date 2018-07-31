import numpy as np
import skimage.morphology as skm


def OpenbyRec(input_im,SE=2):
    """Opening by Reconstruction.

    Arguments:
    input_im: numpy array.

    Returns:
    output_im: numpy array, containing reconstruction from eroded images.
    """
    im_mark = np.copy(input_im)
    im_mark = skm.opening(im_mark,skm.disk(SE)) #replaced erosion by opening
    out_im = skm.reconstruction(im_mark, input_im)
    return out_im

def ClosebyRec(input_im,SE=2):
    """Closing by Reconstruction

    Arguments:
    input_im: numpy array.

    Returns:
    output_im: numpy array, containing reconstruction from close images.
    """
    #c=np.max(input_im)
    #input_im=c-input_im
    im_mark = np.copy(input_im)
    im_mark = skm.closing(im_mark,skm.disk(SE)) 
    #im_mark = skm.opening(im_mark,skm.disk(SE)) #replaced erosion by opening
    out_im = skm.reconstruction(im_mark, input_im, method='erosion')
    #return c-out_im
    return out_im

def dilatation(input_im, SE_scale=2):
    """
    Dilatation by a disk of scale SE_scale.
    Argument:
        input_im: numpy array (nb_rows, nb_images)
    """
    return skm.dilation(input_im,skm.disk(SE_scale))


def AdditiveDecomposition(input_im,levels=4,step=1, init_step=2):
    """Additive Decomposition by Reconstruction
       Version from AMD paper.
    Arguments:
    input_im: numpy array.
    levels: number of levels in the decomposition.
    step: Discretization on the decomposition
    """
    nb_rows, nb_columns = input_im.shape 
    RP=np.zeros((nb_rows, nb_columns, levels))
    RN=np.zeros((nb_rows, nb_columns, levels))
    SE=init_step
    prev_ext = np.copy(input_im)
    prev_antiext = np.copy(input_im)
    for i in range(levels):
        tmp = ClosebyRec(input_im,SE)
        RP[:,:,i]=np.copy(tmp-prev_ext)
        prev_ext = np.copy(tmp)
        tmp = OpenbyRec(input_im,SE)
        RN[:,:,i] = np.copy(prev_antiext-tmp)
        prev_antiext=np.copy(tmp)
        SE=SE+step
    S = (prev_ext + prev_antiext)/2.
    return RP, RN,S


def AdditiveDecomposition2(input_im,levels=4,step=1, init_step=2):
    """Additive Decomposition by Reconstruction. Other version. Not used in the following functions.
    Arguments:
    input_im: numpy array.
    levels: number of levels in the decomposition.
    step: Discretization on the decomposition
    """
    RP=[]
    RN=[]
    SE=init_step
    for i in range(levels):
        out_im=OpenbyRec(input_im,SE)
        RP.append(input_im-out_im)
        out2_im=ClosebyRec(out_im,SE)
        RN.append(out2_im-out_im)
        input_im=out2_im
        SE=SE+step
    return np.array(RP),np.array(RN),input_im

def AMD(X,levels=4,step=1, init_step=1):
    """ AMD with negative and positve residuals combined in one single residual.
        Returns the residual R (nb_images, nb_scales, nb_rows, nb_columns) and the structure S (nb_images, nb_rows, nb_columns)

    Arguments:
    X: a numpy array (n_images, nb_rows, nb_columns).
    levels: number of levels in the decomposition.
    step: Discretization on the decomposition
    """
    nImages = X.shape[0]

    S = np.zeros(np.shape(X))
    R = np.zeros((nImages, np.shape(X)[1], np.shape(X)[2], levels))
    
    for i in range(nImages):
        RP,RN,S[i]=AdditiveDecomposition(X[i],levels=levels,step=step, init_step=init_step)
        R[i] = (RN - RP)/2.
    return R,S

def AMD_in_one_array(X, levels=4,step=1, init_step=1):
    """
    AMD of the set of images X of shape (N_images, N_rows, N_columns)
    Returns the transformed data, in the shape (N_images, N_rows, N_columns, N_AMD_element)
    """
    R,S = AMD(X, levels=levels,step=step, init_step=init_step)
    nb_images, nb_rows, nb_columns = X.shape
    amd = np.concatenate((X.reshape((nb_images, nb_rows, nb_columns, 1)), R, S.reshape((nb_images, nb_rows, nb_columns, 1))), axis=3)
    return amd 
