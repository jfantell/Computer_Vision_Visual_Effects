import numpy as np
import cv2
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import lil_matrix
from scipy.sparse.csc import csc_matrix
import time


def fill_in_A(Mask_Coordinates):
    start = time.time()
    # Initialize A array
    N = len(Mask_Coordinates)

    # # Convert to sparse matrix
    A = lil_matrix((N, N), dtype=np.float32)
    A.setdiag(-4)

    '''
	Create dictionary where keys are white pixels (tuples)
	and values are indices in the White Pixels list
	'''
    coord_to_ind = {}
    for idx, (r, c) in enumerate(Mask_Coordinates):
        coord_to_ind[(r, c)] = idx

    print("Number of white pixels {}".format(len(Mask_Coordinates)))
    for A_row_idx, (r, c) in enumerate(Mask_Coordinates):
        adjacent_pixels_list = [(r, c - 1), (r + 1, c), (r, c + 1), (r - 1, c)]
        for adjacent_pixel in adjacent_pixels_list:
            if adjacent_pixel in coord_to_ind:
                A_col_idx = coord_to_ind[adjacent_pixel]
                A[A_row_idx, A_col_idx] = 1

    # Convert lil to CSC format
    A = csc_matrix(A)

    end = time.time()
    print("Time To Construct A: {}".format(end - start))
    return A


def fill_in_B_regular_composite(S, T, M, Mask_Coordinates):
    start = time.time()
    N = len(Mask_Coordinates)
    b = np.zeros((N,), dtype=np.float32)

    # Filter S
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    del_S = cv2.filter2D(S, cv2.CV_32F, kernel)

    # Filter T
    T = T.copy()
    T[M == 255] = 0
    del_T = cv2.filter2D(T, cv2.CV_32F, kernel)

    # cv2.imwrite("del_S.png",del_S-del_T)
    for idx, (r, c) in enumerate(Mask_Coordinates):
        b[idx] = del_S[r, c] - del_T[r, c]
    end = time.time()
    print("Time To Construct B: {}".format(end - start))
    return b


def fill_in_B_mixed_gradient_composite(S, T, M, Mask_Coordinates):
    start = time.time()
    N = len(Mask_Coordinates)
    b = np.zeros((N,), dtype=np.float32)

    T_border = T.copy()
    T = T.copy()
    T_border[M == 255] = 0

    kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    del_T_border = cv2.filter2D(T_border, cv2.CV_32F, kernel_laplacian)

    # Filter S
    x_kernel = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
    y_kernel = np.array([[0, -1, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.float32)
    del_Sx = cv2.filter2D(S, cv2.CV_32F, x_kernel)
    del_Sy = cv2.filter2D(S, cv2.CV_32F, y_kernel)

    # Filter T
    del_Tx = cv2.filter2D(T, cv2.CV_32F, x_kernel)
    del_Ty = cv2.filter2D(T, cv2.CV_32F, y_kernel)

    # Compute gradient magnitudes
    gm_S = np.sqrt(np.square(del_Sx) + np.square(del_Sy))
    gm_T = np.sqrt(np.square(del_Tx) + np.square(del_Ty))

    # Determine indices where T gm is greater than S gm
    strongest_gm = np.zeros_like(gm_S)
    strongest_gm[gm_T > gm_S] = 1

    # Create vector field G = [Gx, Gy]
    Gx = del_Sx.copy()
    Gy = del_Sy.copy()

    Gx[strongest_gm == 1] = del_Tx[strongest_gm == 1]
    Gy[strongest_gm == 1] = del_Ty[strongest_gm == 1]

    # Compute divergence of vector field
    x_kernel = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    y_kernel = np.array([[0, 0, 0],
                         [0, -1, 0],
                         [0, 1, 0]], dtype=np.float32)
    del_Gx = cv2.filter2D(Gx, cv2.CV_32F, x_kernel)
    del_Gy = cv2.filter2D(Gy, cv2.CV_32F, y_kernel)
    div_GxGy = del_Gx + del_Gy
    # cv2.imwrite("guidance.jpg",div_GxGy)

    for idx, (r, c) in enumerate(Mask_Coordinates):
        b[idx] = div_GxGy[r, c] - del_T_border[r, c]
    end = time.time()
    print("Time To Construct B: {}".format(end - start))
    return b

def fill_in_B_inpainting(S,M,Mask_Coordinates):
    start = time.time()
    N = len(Mask_Coordinates)
    b = np.zeros((N,), dtype=np.float32)

    S = S.copy()
    S[M==255] = 0

    kernel_laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float32)
    del_S = cv2.filter2D(S,cv2.CV_32F,kernel_laplacian)

    for idx, (r,c) in enumerate(Mask_Coordinates):
        b[idx] = 0 - del_S[r,c] #0 guidance vector
    end = time.time()
    print("Time To Construct B: {}".format(end-start))
    return b


def blend_driver(Source_Intensities, T, Mask_Coordinates):
    start = time.time()
    I = T.copy().astype(np.uint8)
    for idx, (r, c) in enumerate(Mask_Coordinates):
        I[r, c] = Source_Intensities[idx]
    end = time.time()
    print("Time To Construct I: {}".format(end - start))
    return I


# Source intensities: list of lists, where each list represents source intensities
# for a particular image channel
def blend(Source_Intensities_3CH, T, Mask_Coordinates):
    three_channel_composites = []
    for channel in range(len(Source_Intensities_3CH)):
        composite = blend_driver(Source_Intensities_3CH[channel], T[:, :, channel], Mask_Coordinates)
        three_channel_composites.append(composite)
    composite_image = cv2.merge(three_channel_composites)
    return composite_image


def solve(A, b):
    start = time.time()
    Source_Intensities_3CH = spsolve(A, b)
    Source_Intensities_3CH[Source_Intensities_3CH > 255] = 255
    Source_Intensities_3CH[Source_Intensities_3CH < 0] = 0
    end = time.time()
    print("Time To Construct Source_Intensities_3CH: {}".format(end - start))
    return Source_Intensities_3CH


def solve_possion_driver(S, T, M, A, Mask_Coordinates, poisson_mode):
    # Create b matrix
    b = None
    if poisson_mode == "Regular":
        b = fill_in_B_regular_composite(S, T, M, Mask_Coordinates)
    elif poisson_mode == "Mixed_Gradients":
        b = fill_in_B_mixed_gradient_composite(S, T, M, Mask_Coordinates)
    else:
        b = fill_in_B_inpainting(S, M, Mask_Coordinates)

    # Solve linear equation
    Source_Intensities_1CH = solve(A, b)
    return Source_Intensities_1CH


def solve_possion(S, T, M, output_dir, poisson_mode="Regular"):
    print("\tComposite\n\n")
    
    if poisson_mode == "Inpaint":
        T = np.zeros((1,3)).reshape(1,1,3)

    # Binarize M
    ret, M = cv2.threshold(M, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    if poisson_mode == "Inpaint":
        kernel = np.ones((3, 3), np.uint8)
    M = cv2.erode(M, kernel, iterations=1)
    R, C = np.where(M == 255)

    # Count number of white pixels
    Mask_Coordinates = list(zip(R, C))
    N = len(Mask_Coordinates)

    # Create A matrix
    A = fill_in_A(Mask_Coordinates)

    # Run the Poission algorithm
    Source_Intensities_3CH = []
    for channel in range(S.shape[2]):
        Source_Intensities_3CH.append(
            solve_possion_driver(S[:, :, channel], T[:, :, channel], M, A, Mask_Coordinates, poisson_mode))
    return (Source_Intensities_3CH, Mask_Coordinates)
