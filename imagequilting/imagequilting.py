import cv2 as cv
import math
import random
import numpy
from scipy import signal

VERTICAL_OVERLAP = 0
HORIZONTAL_OVERLAP = 1
L_OVERLAP = 2

"""
    Main algorithm that implements the image quilting algorithm detailed in
    Efros and Friedman's paper. Image is saved to /output directory

    Accepts the following parameters:
    :param inputfile (string) the name of the input texture
    :param patchsize (int)    the size of the patches, cannot be bigger than the entire image
    :param factor    (float)  the increase in size of the original image
    :param overlap   (float)  the overlap size between blocks 
    :param tolerance (float)  tolerance for edge similarity

"""
def image_quilting(inputfile, patchsize=40, factor=1.5, overlap=.25, tolerance=0.01):
    # Open image
    Io = cv.imread(inputfile, cv.IMREAD_UNCHANGED)

    # Get the current dimensions
    # height = # rows, width = # columns
    imheight, imwidth, imdim = Io.shape

    # Checking parameters
    if patchsize > (imwidth - patchsize) or patchsize > (imheight - patchsize) or patchsize <= 1:
        print("Patch size must be smaller than image dimensions minus patchsize, and greater than one")
        return

    if factor <= 1.0:
        print("Resize factor should be greater than 1")
        return

    if overlap < 0.0 or overlap > 1.0:
        print("Overlap should be between 0.0 and 1.0")
        return

    overlap_distance = int(round(patchsize * overlap))

    # Get the new width and height for output image
    resized_height, resized_width = round(imheight * factor), round(imwidth * factor)

    # Get the number of patches per row and per column
    p_row, p_col = get_num_patches(resized_height, resized_width, patchsize, overlap_distance)

    # Initialize Is, the output image, to be a random patch
    Is = init_is(Io, (resized_width, resized_height, imdim), patchsize)

    for m in range(0, p_row):
        for n in range(0, p_col):
            if m == 0 and n == 0:
                continue

            # Deciding overlap type
            overlap_type = L_OVERLAP
            if m == 0:
                overlap_type = VERTICAL_OVERLAP
            elif n == 0:
                overlap_type = HORIZONTAL_OVERLAP

            start_x = m * (patchsize - overlap_distance)
            start_y = n * (patchsize - overlap_distance)

            # Get the overlay region for the next patch in Is
            Pold = get_next_p_old_for_coords(Is, start_x, start_y, patchsize, overlap_distance, overlap_type)
            Pin = select_compatible_patch(Io, Pold, patchsize, overlap_distance, overlap_type, tolerance)

            '''tokens = inputfile.split("/")
            outname = "output/Pin.jpg"
            cv.imwrite(outname, Pin)
            exit(0)'''

            Pnew = compute_min_boundary_cut(Pold, Pin, patchsize, overlap_distance, imdim, overlap_type)

            # Copy over selected patch
            for row in range(0, patchsize):
                for col in range(0, patchsize):
                    if row + start_x >= Is.shape[0] or col + start_y >= Is.shape[1]:
                        continue

                    Is[row + start_x][col + start_y] = Pnew[row][col]

        '''tokens = inputfile.split("/")
        outname = "output/" + tokens[len(tokens) - 1]
        cv.imwrite(outname, Is)
        exit(0)'''

    # Write result to output directory
    tokens = inputfile.split("/")
    outname = "output/" + tokens[len(tokens) - 1]
    cv.imwrite(outname, Is)


"""
    Helper function that determine how many patches per row and 
    column there should be given an image, a resize factor, and the given patchsize

    :param resized_dim (float[])  resized dimensions of the image
    :param patchsize   (int)    the size of the patches, cannot be bigger than the entire image
    :param factor      (float)  the increase in size of the original image

    :return patches_row, patches_col (int, int) patches per row, patches per column
"""


def get_num_patches(resized_height, resized_width, patchsize, overlap_distance):
    # Number of patches is the re-sized dimensions divided by patchsize - overlap_distance
    return math.ceil(resized_height / (patchsize - overlap_distance)), \
           math.ceil(resized_width / (patchsize - overlap_distance))


"""
    Helper function that will retrieve a random patch from an original image
    with the given patch size

    Randomly selects a row and column to start from, which will be the top left
    corner of the patch

    :param Io        (image) Original image
    :param patchsize (int)   size of each square patch
"""


def get_random_patch(Io, patchsize):
    # Get the current dimensions
    imheight, imwidth = Io.shape[:2]

    # Randomly generate a (row, col) to start from
    rand_row, rand_col = random.randint(0, imheight - patchsize), random.randint(0, imwidth - patchsize)

    # Return patch
    return Io[rand_row:(rand_row + patchsize), rand_col:(rand_col + patchsize)]


"""
    Helper function that will generalized the initialized Is by 
    randomly quilting together patches of images from the initial image Io
    
    :param Io          (ndarray) the original image
    :param resized_dim (int tuple) the resized dimensions for the new image
    :param patchsize   (int) Wp, the patch size
"""


def init_is(Io, resized_dim, patchsize):
    # Initialize new image
    Is = numpy.zeros(resized_dim)

    # Grab a random patch
    patch = get_random_patch(Io, patchsize)

    # Copy over patch information into 0, 0
    for row in range(0, patchsize):
        for col in range(0, patchsize):
            # Copy over patch
            Is[row][col] = patch[row][col]

    return Is

"""
    Helper function that gets from Is the overlap region that our algorithm will have to 
    match. 
    
    :param Is  (ndarray) the synthesized image
    :param row (int)     the row that we will be overlaying in Is with a new patch
    :param col (int)     the column that we will be overlaying in Is with a new patch
    :param patchsize        (int) Wp, the patch size
    :param overlap_distance (int) Wo, the overlap distance
    :param overlap_type     (int) HORIZONTAL_OVERLAP, VERTICAL_OVERLAP or L_OVERLAP
"""
def get_next_p_old_for_coords(Is, row, col, patchsize, overlap_distance, overlap_type):
    full_patch = numpy.zeros((patchsize, patchsize, Is.shape[2]))
    for i in range(0, patchsize):
        for j in range(0, patchsize):
            if i + row >= Is.shape[0] or j + col >= Is.shape[1]:
                continue

            full_patch[i][j] = Is[i + row][j + col]

    mask = get_overlap_for_type(patchsize, overlap_distance, overlap_type, Is.shape[2])
    return numpy.multiply(full_patch, mask)


"""
    Helper function that will return the horizontal-shaped overlap mask

    :param patchsize        (int) Wp, the patch size
    :param overlap_distance (int) Wo, the overlap distance
    :param dims             (int) dimensionality of mask, defaults to 1D
"""


def get_horizontal_overlap_mask(patchsize, overlap_distance, dims):
    mask = numpy.zeros((patchsize, patchsize, dims))
    for row in range(0, overlap_distance):
        for col in range(0, patchsize):
            if dims == 1:
                mask[row][col] = 1
            else:
                mask[row][col] = numpy.ones((dims,))
    return mask


"""
    Helper function that will return the vertical-shaped overlap mask

    :param patchsize        (int) Wp, the patch size
    :param overlap_distance (int) Wo, the overlap distance
    :param dims             (int) dimensionality of mask, defaults to 1D
"""


def get_vertical_overlap_mask(patchsize, overlap_distance, dims):
    mask = numpy.zeros((patchsize, patchsize, dims))
    for row in range(0, patchsize):
        for col in range(0, overlap_distance):
            if dims == 1:
                mask[row][col] = 1
            else:
                mask[row][col] = numpy.ones((dims,))
    return mask


"""
    Helper function that will return the L-shaped overlap mask

    :param patchsize        (int) Wp, the patch size
    :param overlap_distance (int) Wo, the overlap distance
    :param dims             (int) dimensionality of mask, defaults to 1D
"""


def get_L_overlap_mask(patchsize, overlap_distance, dims):
    mask = numpy.zeros((patchsize, patchsize, dims))
    for row in range(0, overlap_distance):
        for col in range(0, patchsize):
            if dims == 1:
                mask[row][col] = 1
            else:
                mask[row][col] = numpy.ones((dims,))

    for row in range(0, patchsize):
        for col in range(0, overlap_distance):
            if dims == 1:
                mask[row][col] = 1
            else:
                mask[row][col] = numpy.ones((dims,))
    return mask


"""
    Helper function that will return the correct mask for the given overlap type

    :param patchsize        (int) Wp, the patch size
    :param overlap_distance (int) Wo, the overlap distance
    :param overlap_type     (int) HORIZONTAL_OVERLAP, VERTICAL_OVERLAP or L_OVERLAP
    :param dims             (int) dimensionality of mask, defaults to 1D
"""


def get_overlap_for_type(patchsize, overlap_distance, overlap_type, dims=1):
    if overlap_type == HORIZONTAL_OVERLAP:
        return get_horizontal_overlap_mask(patchsize, overlap_distance, dims)
    elif overlap_type == VERTICAL_OVERLAP:
        return get_vertical_overlap_mask(patchsize, overlap_distance, dims)
    else:
        return get_L_overlap_mask(patchsize, overlap_distance, dims)


"""
    Helper function that will select the most compatible patch with the given random patch
    sampled from the original image

    :param Io               (image)   Original image
    :param oldpatch         (ndarray) random patch chosen
    :param patchsize        (int)     size of each square patch
    :param overlap_distance (int) size of overlap gap between patches
    :param tolerance        (float) tolerance of similarity between viable patches
"""


def select_compatible_patch(Io, oldpatch, patchsize, overlap_distance, overlap_type, tolerance):
    # Get the current dimensions
    imheight, imwidth = Io.shape[:2]

    # initialize D-image dimensions (distance image)
    d_rows = imheight - patchsize
    d_cols = imwidth - patchsize

    # Distance image is found by finding the SSD if the patch is started at each point
    # This is the same of performing template matching with the overlapping regions
    D = compute_D(Io, oldpatch, patchsize, overlap_distance, overlap_type)
    D = numpy.abs(D)

    # Find the minimum distance. Will be used with tolerance to filter out patches
    d_min = float("inf")
    dmin_x = dmin_y = 0
    for row in range(0, d_rows):
        for col in range(0, d_cols):
            # All SSD errors equal to zero (aka identical patches) are set to infinity to avoid repetition
            if D[row][col] == 0:
                D[row][col] = float("inf")

            if D[row][col] < d_min:
                d_min = D[row][col]
                dmin_x = row
                dmin_y = col

    # Calculate minimum value
    minimum_val = (1 + tolerance) * d_min

    # Get the set of all viable patches whose di
    viable_set = []
    for row in range(0, d_rows):
        for col in range(0, d_cols):
            if D[row][col] < minimum_val:
                viable_set.append((row, col))

    # Randomly choose a patch to use in the viable set of patches
    if len(viable_set) == 0:
        m, n = dmin_x, dmin_y
    else:
        m, n = viable_set[random.randint(0, len(viable_set) - 1)]
    print("Picked tile (", m, ", ", n, ")", "  out of ", len(viable_set), " candidates. Best error = ", d_min)
    return Io[m:(m + patchsize), n:(n + patchsize)]


"""
    Helper function that will compute the distance measure for the new patch given at coords

    :param Io               (image)   Original image
    :param oldpatch         (ndarray) random patch chosen
    :param patchsize        (int)     size of each square patch
    :param overlap_distance (int) size of overlap gap between patches
    :param overlap_type     (int) HORIZONTAL_OVERLAP, VERTICAL_OVERLAP or L_OVERLAP
"""


def compute_D(Io, oldpatch, patchsize, overlap_distance, overlap_type):
    # Get the current dimensions
    imheight, imwidth = Io.shape[:2]

    # initialize distance_image dimensions
    d_rows = imheight - patchsize
    d_cols = imwidth - patchsize

    distance_image = numpy.zeros((d_rows, d_cols, 1))
    mask = get_overlap_for_type(patchsize, overlap_distance, overlap_type, Io.shape[2])
    for m in range(0, patchsize - overlap_distance):
        for n in range(0, patchsize - overlap_distance):
            distance_image[m][n] = ssd(oldpatch, Io[m:(m + patchsize), n:(n + patchsize)], mask)
    return distance_image

"""
    Helper function that returns the ssd of the two patches for a given overlap region

    D[m][n] = sum from i, j = 0 (Q(i, j) - (Pold(i, j) + Io[i + m][j + n])^2)

    :param oldpatch         (ndarray) random patch chosen
    :param inpatch          (ndarray) prospective patch to compare
    :param mask             (ndarray) overlap region representation
"""

def ssd(oldpatch, inpatch, mask):
    #TODO: Originally sum?
    return numpy.multiply(mask, numpy.square(numpy.subtract(oldpatch, inpatch))).mean()

"""
    Calculates the minimum boundary cut and returns the patch that will be transposed onto the 
    final image

    :param oldpatch         (ndarray) random patch chosen
    :param inpatch          (ndarray) patch that will overlap
    :param patchsize        (int)     size of each square patch
    :param overlap_distance (int)     size of overlap gap between patches
    :param dims             (int)     dimensions of image
    :param overlap_type     (int)     HORIZONTAL_OVERLAP, VERTICAL_OVERLAP or L_OVERLAP
"""


def compute_min_boundary_cut(oldpatch, inpatch, patchsize, overlap_distance, dims, overlap_type):
    mask = numpy.zeros((patchsize, patchsize, dims))
    if overlap_type == HORIZONTAL_OVERLAP:
        Eh = calculate_horizontal_path(oldpatch, inpatch, patchsize, overlap_distance)
        mask = fill_in_horizontal_path(Eh, mask, patchsize, overlap_distance, dims)
    elif overlap_type == VERTICAL_OVERLAP:
        Ev = calculate_vertical_path(oldpatch, inpatch, patchsize, overlap_distance)
        mask = fill_in_vertical_path(Ev, mask, patchsize, overlap_distance, dims)
    else:
        Ev = calculate_vertical_path(oldpatch, inpatch, patchsize, overlap_distance)
        Eh = calculate_horizontal_path(oldpatch, inpatch, patchsize, overlap_distance)
        mask = fill_in_vertical_path(Ev, mask, patchsize, overlap_distance, dims)
        mask = fill_in_horizontal_path(Eh, mask, patchsize, overlap_distance, dims)

    return (mask * oldpatch) + ((1 - mask) * inpatch)


"""
    Helper function that calculates the minimum vertical path for a given set of patches that will
    overlap

    :param oldpatch         (ndarray) random patch chosen
    :param inpatch          (ndarray) patch that will overlap
    :param patchsize        (int)     size of each square patch
    :param overlap_distance (int)     size of overlap gap between patches
"""


def calculate_vertical_path(oldpatch, inpatch, patchsize, overlap_distance):
    Ev = numpy.zeros((patchsize, patchsize))
    for row in range(patchsize - 1, -1, -1):
        for col in range(0, overlap_distance):
            currError = sse_error(inpatch[row][col], oldpatch[row][col])

            # If the row + 1 is out of bounds, then we are filling in the first layer
            if row + 1 >= patchsize:
                Ev[row][col] = currError
                continue

            minVal = Ev[row + 1][col]

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if col - 1 >= 0:
                minVal = min(minVal, Ev[row + 1][col - 1])

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if col + 1 < overlap_distance:
                minVal = min(minVal, Ev[row + 1][col + 1])

            # Cost for this value is the minVal + currError
            Ev[row][col] = currError + minVal
    return Ev


"""
    Helper function that calculates the minimum horizontal path for a given set of patches that will
    overlap

    :param oldpatch         (ndarray) random patch chosen
    :param inpatch          (ndarray) patch that will overlap
    :param patchsize        (int)     size of each square patch
    :param overlap_distance (int)     size of overlap gap between patches
"""


def calculate_horizontal_path(oldpatch, inpatch, patchsize, overlap_distance):
    Eh = numpy.zeros((patchsize, patchsize))
    for col in range(patchsize - 1, -1, -1):
        for row in range(overlap_distance - 1, -1, -1):
            currError = sse_error(inpatch[row][col], oldpatch[row][col])

            # If the row + 1 is out of bounds, then we are filling in the first layer
            if col + 1 >= patchsize:
                Eh[row][col] = currError
                continue

            minVal = Eh[row][col + 1]

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if row - 1 >= 0:
                minVal = min(minVal, Eh[row - 1][col + 1])

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if row + 1 < overlap_distance:
                minVal = min(minVal, Eh[row + 1][col + 1])

            # Cost for this value is the minVal + currError
            Eh[row][col] = currError + minVal
    return Eh


"""
    Helper function that will fill in the mask of the minimum path such that
    all pixels to the left/top of the cut are 1 and all others are
    0
"""


def fill_in_vertical_path(Ev, mask, patchsize, overlap_distance, dims):
    # Step 4.a Trace the vertical of path starting from (i, j) and going down
    # Find the starting point of the vertical path
    currRow = 0
    currCol = 0
    currMin = Ev[currRow][currCol]
    for col in range(1, overlap_distance):
        if currMin > Ev[currRow][col]:
            currCol = col
            currMin = Ev[currRow][col]

    while True:
        # Set all points to the left of currCol as 1, others as 0
        for col in range(0, currCol):
            mask[currRow][col] = numpy.ones((dims,))

        # If we are at the last row, then exit the loop
        if currRow + 1 >= patchsize:
            break

        # Otherwise, update the current row and the current column

        leftVal = float("inf")
        middleVal = Ev[currRow + 1][currCol]
        rightVal = float("inf")

        # Check to make sure not out of bounds. If not, set rightVal
        if currCol - 1 >= 0:
            leftVal = Ev[currRow + 1][currCol - 1]

        # Check to make sure not out of bounds. If not, set leftval
        if currCol + 1 < overlap_distance:
            rightVal = Ev[currRow + 1][currCol + 1]

        # Curr col is set to lowest value's col, curr row is decremented

        if leftVal < rightVal and leftVal < middleVal:
            currCol = currCol - 1
        elif rightVal < leftVal and rightVal < middleVal:
            currCol = currCol + 1

        currRow = currRow + 1

    return mask


"""
    Helper function that will fill in the mask of the minimum path such that
    all pixels to the left/top of the cut are ____(one?)____ and all others are
    ___(zero?)____
"""


def fill_in_horizontal_path(Eh, mask, patchsize, overlap_distance, dims):
    # Trace the horizontal path
    # Find the starting point of the horizontal path
    currRow = 0
    currCol = 0
    currMin = Eh[currRow][currCol]
    for row in range(1, overlap_distance):
        if currMin > Eh[row][currCol]:
            currRow = row
            currMin = Eh[row][currCol]

    while True:

        # Set all points above the curr row as one
        for row in range(0, currRow):
            mask[row][currCol] = numpy.ones((dims,))

        # If we are at the last column, then exit the loop
        if currCol + 1 >= patchsize:
            break

        # Otherwise, update the current row and the current column
        leftVal = float("inf")
        middleVal = Eh[currRow][currCol + 1]
        rightVal = float("inf")

        # Check to make sure not out of bounds. If not, set rightVal
        if currRow - 1 > -1:
            leftVal = Eh[currRow - 1][currCol + 1]

        # Check to make sure not out of bounds. If not, set leftval
        if currRow + 1 < overlap_distance:
            rightVal = Eh[currRow + 1][currCol + 1]

        # Curr row is set to lowest value's row, curr column is decremented

        if leftVal < rightVal and leftVal < middleVal:
            currRow = currRow + 1
        elif rightVal < leftVal and rightVal < middleVal:
            currRow = currRow - 1

        currCol = currCol + 1
    return mask


"""
    Helper function that will compute the sum of squared error
    for two given input vectors
"""


def sse_error(v1, v2):
    return numpy.square(numpy.subtract(v1, v2)).mean()


"""
    Helper function that computes the L2 Norm of 
    a vector value
"""


def l2_norm(v1):
    squared_sum = 0
    for i in range(0, v1.shape[0]):
        squared_sum += math.pow(v1[i], 2)
    return math.sqrt(squared_sum)


def main():
    image_quilting("images/bubbly/bubbly_0054.jpg", factor=1.1)


main()