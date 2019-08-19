
import numpy as np

# ######functions for generating filter
def make_gaussian(center,sig,n_pix):
    """
    Make a picture of a circular gaussian blob.
    center is the center of the blob in pixels. center of image is (0,0)
    
    sig is one std. of the gaussian (pixels)
    
    n_pix is the size of the picture of the gaussian blob. i.e., output will be an 2D array that is n_pix-by-n_pix
    """
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    
    [Xm, Ym] = np.meshgrid(range(pix_min,pix_max), range(pix_min,pix_max));  
    
    x0 = center[0]
    y0 = center[1]
    
    Z = (1. / 2*np.pi*sig**2)
    
    return Z *np.exp(-((Xm-x0)**2 + (Ym-y0)**2) / (2*sig**2))
  
  
def make_2D_sinewave(freq, theta, phase, n_pix):
    '''
    
    freq is cycles/image
    
    theta is in radians
    
    phase is in radians (0 pi)
    
    center is (x,y) in pixel coordinates
    
    n_pix is size of the kernel in pixels
    
    '''
    vec = np.array([np.sin(theta), np.cos(theta)]).reshape((2,1))
    
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    [Xm, Ym] = np.meshgrid(range(pix_min,pix_max), range(pix_min,pix_max));
    proj = np.array([Xm.ravel(), Ym.ravel()]).T.dot(vec)
    Dt = np.sin(proj/n_pix*freq*2*np.pi+phase)              # compute proportion of Xm for given orientation\
    Dt = Dt.reshape(Xm.shape)
    return Dt
  
  
def make_gabor(freq, theta, phase, center, sig, n_pix):
  return make_2D_sinewave(freq,theta,phase,n_pix)*make_gaussian(center, sig,n_pix)

def make_complex_gabor(freq,theta, center,sig,n_pix):
    '''
    make_complex_gabor(freq,theta, center,sig,n_pix)
    freq is spatial frequency in cycles/image
    theta is orientation in radians
    center is (x,y) in pixel coordinates. center of image is (0,0)
    sig is one std of the gaussian envelope (pixels)
    n_pix is size of the kernel in pixels
    
    '''
    phase = 0
    on_gabor = make_gabor(freq, theta, phase, center, sig, n_pix)
    phase = np.pi/2.
    off_gabor = make_gabor(freq, theta, phase, center, sig, n_pix)
    return off_gabor + 1j*on_gabor

def make_a_ripple(freq, phase_off, center, n_pix):
    '''
    freq is cycles/image
    
    phase_off is in radians (0 pi)
    
    center is (x,y) in pixel coordinates
    
    n_pix is size of the kernel in pixels
    
    '''
    
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    [Xm, Ym] = np.meshgrid(range(pix_min,pix_max), range(pix_min,pix_max));  
    Dt = np.sin(np.sqrt((Xm-center[0])**2+(Ym-center[1])**2)/n_pix*freq*2*np.pi+phase_off)              # compute proportion of Xm for given orientation
    return Dt

def complex_ripple_filter(freq,center,fwhm,n_pix):
    '''
    freq ~ cyc/image
    
    center ~ (x,y) in pixels, origin is center of picture
    
    fwhm ~ full-width half max of Gaussian envelope = diameter in pixels at half-max
    
    n_pix ~ size of square image in pixels
    
    '''
    phase = 0
    on_ripple = make_a_ripple(freq,phase,center,n_pix)*make_gaussian(center,fwhm,n_pix)
    phase = np.pi/2.
    off_ripple = make_a_ripple(freq,phase,center,n_pix)*make_gaussian(center,fwhm,n_pix)
    return off_ripple + 1j*on_ripple
  
def compute_grid_corners(n_pix, kernel_size,boundary_condition=0):
    '''
    compute_grid_corners(n_pix, kernel_size)
    return corners of placement grid in image of size n_pix given kernel_size (radius)
    and a boundary_condition. boundary_condition >= 0 is the distance of the grid corners from the corners
    of the image in units of kernel_size. So boundary_condition=0 means corners of grid are corners of the 
    image, while boundary_condition = 1 would mean the grid corners are one kernel_size away from the corners
    of the image.
    [left, right, top, bottom]
    '''
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    
    ks = kernel_size*boundary_condition
    return np.array([pix_min+ks,pix_max-ks,pix_min+ks,pix_max-ks])

def compute_grid_spacing(kernel_size,fraction_of_kernel_size):
    '''
    compute_grid_spacing(kernel_size,fraction_of_kernel_size)
    
    returns integer distance in pixels between each kernel.
    spacing =  int(kernel_size*fraction_of_kernel_size)
   
    '''
    
    gs = int(kernel_size*fraction_of_kernel_size)
    return gs

def construct_placement_grid(grid_corners, grid_spacing):
    '''
    X,Y = construct_placement_grid(grid_corners, grid_spacing)
    given [left,right,top,bottom] corners of the grid, and an integer pixel spacing,
    return a grid of kernel placements.
    '''
    num0 = int((grid_corners[1]-grid_corners[0])/grid_spacing)
    num1= int((grid_corners[3]-grid_corners[2])/grid_spacing)
    X,Y = np.meshgrid(np.linspace(grid_corners[0],grid_corners[1],num=num0,endpoint=True),
		  np.linspace(grid_corners[2],grid_corners[3],num=num1,endpoint=True))
    return X,Y

def construct_kernel_set(freq,kernel_size,n_pix,kernel_spacing, boundary_condition=0):
    '''
    construct_kernel_set(freq,kernel_size,n_pix,kernel_spacing, boundary_condition=0)
    
    freq ~ cyc/n_pix
    
    kernel_size ~ pix. the fwhm of the gaussian envelope, effectively the kernel radius.
    
    n_pix ~ pixels per side of square image
    
    kernel_spacing ~ in units of kernel_size. 
    
    boundary_condition ~ >= 0. boundary_condition >= 0 distance of grid corners from corners
    of image in units of kernel_size.
    
    returns:
    
    kernel_set  = 3D numpy array of complex ripple filters ~ [number_of_filters] x [n_pix] x [n_pix]
    
    iter_x, iter_y ~ centers of the filters. number of filters = len(iter_x) = len(iter_y)
    
    
    
    '''
    grid_corners = compute_grid_corners(n_pix, kernel_size,boundary_condition)
    grid_spacing = compute_grid_spacing(kernel_size,kernel_spacing)
    gridX, gridY = construct_placement_grid(grid_corners, grid_spacing)
    iter_x = np.ravel(gridX)
    iter_y = np.ravel(gridY)
    kernel_set = np.zeros((len(iter_x),n_pix,n_pix)).astype(complex)
    count = 0    
    print 'constructing %d filters' %(len(iter_x))
    for x,y in zip(iter_x,iter_y):
        kernel_set[count,:,:] = complex_ripple_filter(freq, (x,y),kernel_size,n_pix)
        count += 1
    return kernel_set,iter_x,iter_y
  
def make_kernel_grid(freq,kernel_size,n_pix,placement_grid):
    '''
    make_kernel_grid(freq,kernel_size,n_pix,placement_grid)
    
    freq ~ cyc/n_pix
    
    kernel_size ~ pix. the fwhm of the gaussian envelope, effectively the kernel radius.
    
    n_pix ~ pixels per side of square image
    
    placement_grid = (X,Y) grid of kernel centers, as from meshgrid
   
    return:
      kernel_set  = 3D numpy array of complex ripple filters ~ [number_of_filters] x [n_pix] x [n_pix]
    
    '''
    iter_x = np.ravel(placement_grid[0])
    iter_y = np.ravel(placement_grid[1])

    kernel_set = np.zeros((len(iter_x),n_pix,n_pix)).astype(complex)
    count = 0    
    print 'constructing %d filters' %(len(iter_x))
    for x,y in zip(iter_x,iter_y):
        kernel_set[count,:,:] = complex_ripple_filter(freq, (x,y),kernel_size,n_pix)
        count += 1
    return kernel_set