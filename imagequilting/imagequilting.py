import cv2 as cv
import math
import random
import numpy
import sys

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
def image_quilting(inputfile, patchsize=40, factor=1.5, overlap=.25, tolerance=.1):
    # Open image
    Io = cv.imread(inputfile, cv.IMREAD_UNCHANGED)

    # Get the current dimensions
    imwidth, imheight, imdim = Io.shape

    # Checking parameters
    if patchsize > imwidth or patchsize > imheight or patchsize == 0:
        print("Patch size must be smaller than image dimensions, and greater than zero")
        return

    if factor <= 1.0:
        print("Resize factor should be greater than 1")
        return

    if overlap < 0.0 or overlap > 1.0:
        print("Overlap should be between 0.0 and 1.0")
        return

    overlap_distance = int(round(patchsize * overlap))

    # Get the new width and height for output image
    resized_width, resized_height = round(imwidth * factor), round(imheight * factor)

    # Get the number of patches per row and per column
    p_row, p_col = get_num_patches((resized_width, resized_height), patchsize, overlap_distance)

    # Initialize Is, the output image, to be random patches
    Pold = get_random_patch(Io, patchsize)
    Is = init_is(Pold, (resized_width, resized_height, imdim), patchsize)

    num_inserted = 0
    for m in range(0, p_row):
        if num_inserted > 1:
            break
        for n in range(0, p_col):
            if num_inserted > 1:
                break

            if m == 0 and n == 0:
                continue

            start_x = m * (patchsize - overlap_distance)
            start_y = n * (patchsize - overlap_distance)

            Pin = select_compatible_patch(Io, Pold, patchsize, overlap_distance, tolerance)
            Pnew = compute_min_boundary_cut(Pold, Pin, patchsize, overlap_distance)

            # Copy over selected patch
            for row in range(0, patchsize):
                for col in range(0, patchsize):
                    if row + start_x >= Is.shape[0] or col + start_y >= Is.shape[1]:
                        continue

                    Is[row + start_x][col + start_y] = Pnew[row][col]

            num_inserted += 1

            Pold = Pnew

    # Write result to output directory
    cv.imwrite("output/brick.jpg", Is)

"""
    Helper function that determine how many patches per row and 
    column there should be given an image, a resize factor, and the given patchsize
    
    :param resized_dim (float[])  resized dimensions of the image
    :param patchsize   (int)    the size of the patches, cannot be bigger than the entire image
    :param factor      (float)  the increase in size of the original image
    
    :return patches_row, patches_col (int, int) patches per row, patches per column
"""
def get_num_patches(resized_dim, patchsize, overlap_distance):
    # Number of patches is the re-sized dimensions divided by patchsize - overlap_distance
    return math.ceil(resized_dim[1] / (patchsize - overlap_distance)), \
           math.ceil(resized_dim[0] / (patchsize - overlap_distance))

"""
    Helper function that will generalized the initialized Is by 
    randomly quilting together patches of images from the initial image Io
"""
def init_is(patch, resized_dim, patchsize):
    # Initialize new image
    Is = numpy.zeros(resized_dim)

    # Copy over patch information into 0, 0
    for row in range(0, patchsize):
        for col in range(0, patchsize):
            # Copy over patch
            Is[row][col] = patch[row][col]

    return Is

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
    imwidth, imheight = Io.shape[:2]

    # Randomly generate a (row, col) to start from
    rand_row, rand_col = random.randint(0, imwidth - patchsize), random.randint(0, imheight - patchsize)

    # Return patch
    return Io[rand_row:(rand_row + patchsize), rand_col:(rand_col + patchsize)]

"""
    Helper function that will select the most compatible patch with the given random patch
    sampled from the original image
    
    :param Io               (image)   Original image
    :param oldpatch         (ndarray) random patch chosen
    :param patchsize        (int)     size of each square patch
    :param overlap_distance (int) size of overlap gap between patches
    :param tolerance        (float) tolerance of similarity between viable patches
"""
def select_compatible_patch(Io, oldpatch, patchsize, overlap_distance, tolerance):
    # Get the current dimensions
    imwidth, imheight = Io.shape[:2]

    # initialize D-image dimensions (distance image)
    d_rows = imwidth - patchsize
    d_cols = imheight - patchsize

    # Distance image is found by finding the SSD if the patch is started at each point
    # This is the same of performing template matching with the overlapping regions
    D = compute_D(Io, oldpatch, patchsize, overlap_distance)

    # Find the minimum distance. Will be used with tolerance to filter out patches
    d_min = (0, 0)
    for row in range(0, d_rows):
        for col in range(0, d_cols):
            # All SSD errors smaller than one are set to 1
            if D[row][col] < 1:
                D[row][col] = 1

            if D[d_min[0]][d_min[1]] > D[row][col]:
                d_min = (row, col)

    # Calculate minimum value
    minimum_val = (1 + tolerance) * D[d_min[0]][d_min[1]]

    # Get the set of all viable patches whose di
    viable_set = []
    for row in range(0, d_rows):
        for col in range(0, d_cols):
            if D[row][col] < minimum_val:
                viable_set.append((row, col))

    # Randomly choose a patch to use in the viable set of patches
    m, n = viable_set[random.randint(0, len(viable_set))]
    return Io[m:(m + patchsize), n:(n + patchsize)]

"""
    Helper function that will compute the distance measure for the new patch given at coords
"""
def compute_D(Io, oldpatch, patchsize, overlap_distance):
    IoGray = cv.cvtColor(Io, cv.COLOR_RGB2GRAY)

    oldpathConverted = numpy.uint8(oldpatch)
    patchGray = cv.cvtColor(oldpathConverted, cv.COLOR_RGB2GRAY)

    mask = numpy.zeros((patchsize, patchsize))
    for row in range(patchsize - overlap_distance, patchsize):
        for col in range(0, patchsize):
            mask[row][col] = 1

    for row in range(0, patchsize):
        for col in range(patchsize - overlap_distance, patchsize):
            mask[row][col] = 1

    return cv.matchTemplate(image=IoGray, templ=patchGray, mask=mask, method=cv.TM_SQDIFF)

"""
    Helper function that computes the L2 Norm of 
    a vector value
"""
def l2_norm(v1):
    squared_sum = 0
    for i in range(0, v1.shape[0]):
        squared_sum += math.pow(v1[i], 2)
    return math.sqrt(squared_sum)

def compute_min_boundary_cut(oldpatch, inpatch, patchsize, overlap_distance):
    Ev = numpy.zeros((patchsize, patchsize))
    Eh = numpy.zeros((patchsize, patchsize))

    # Step 1: Calculate the vertical path, starting from the bottom
    for row in range(patchsize - 1, -1, -1):
        for col in range(patchsize - overlap_distance, patchsize):
            currError = sse_error(inpatch[row][col], oldpatch[row][col])

            # If the row + 1 is out of bounds, then we are filling in the first layer
            if row + 1 >= patchsize:
                Ev[row][col] = currError
                continue

            minVal = Ev[row + 1][col]

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if col - 1 >= patchsize - overlap_distance:
                minVal = min(minVal, Ev[row + 1][col - 1])

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if col + 1 < patchsize:
                minVal = min(minVal, Ev[row + 1][col + 1])

            # Cost for this value is the minVal + currError
            Ev[row][col] = currError + minVal

    # Step 2: Calculate the horizontal path
    for col in range(patchsize - 1, -1, -1):
        for row in range(patchsize - 1, patchsize - overlap_distance - 1, -1):
            currError = sse_error(inpatch[row][col], oldpatch[row][col])

            # If the row + 1 is out of bounds, then we are filling in the first layer
            if col + 1 >= patchsize:
                Eh[row][col] = currError
                continue

            minVal = Eh[row][col + 1]

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if row - 1 >= patchsize - overlap_distance:
                minVal = min(minVal, Eh[row - 1][col + 1])

            # Check to make sure not out of bounds. If not, reset minVal to minimum of values
            if row + 1 < patchsize:
                minVal = min(minVal, Eh[row + 1][col + 1])

            # Cost for this value is the minVal + currError
            Eh[row][col] = currError + minVal

    # Step 3: Find the optimal join point in the overlap
    minOverlap = None
    i = j = 0
    for row in range(patchsize - 1, patchsize - overlap_distance - 1, -1):
        for col in range(patchsize - overlap_distance, patchsize):
            currError = sse_error(inpatch[row][col], oldpatch[row][col])
            if minOverlap is None or (Ev[row][col] + Eh[col][row] - currError) < minOverlap:
                minOverlap = Ev[row][col] + Eh[col][row] - currError
                i = row
                j = col

    print("(", i, ", ", j, ")")
    # Step 4: Compute binary mask M that is 1 left/on top of cut and 0 otherwise
    mask = numpy.zeros(oldpatch.shape)

    # Step 4.a Trace the horizontal of y starting from (i, j)
    currRow = i
    currCol = j
    while currCol > -1:

        # Set all points above the curr row as one
        row = currRow
        while row > (patchsize - overlap_distance) - 1:
            mask[row][currCol] = 1
            row -= 1

        # Update the current row and the current column
        leftVal = float("inf")
        middleVal = Eh[currRow][currCol - 1]
        rightVal = float("inf")

        # Check to make sure not out of bounds. If not, set rightVal
        if currRow - 1 > -1:
            rightVal = Eh[currRow - 1][currCol - 1]

        # Check to make sure not out of bounds. If not, set leftval
        if currRow + 1 < overlap_distance:
            leftVal = Eh[currRow + 1][currCol - 1]

        # Curr row is set to lowest value's row, curr column is decremented

        if leftVal < rightVal and leftVal < middleVal:
            currRow = currRow + 1
        elif rightVal < leftVal and rightVal < middleVal:
            currRow = currRow - 1

        currCol = currCol - 1

    # Step 4.b
    currRow = i
    currCol = j
    while (currRow > -1):
        # Set all points to the left of currCol as 1, others as 0
        for col in range(overlap_distance - patchsize, patchsize):
            if col <= currCol:
                mask[currRow][col] = 1
            else:
                mask[currRow][col] = 0

        # Update the current row and the current column

        leftVal = float("inf")
        middleVal = Eh[currRow - 1][currCol]
        rightVal = float("inf")

        # Check to make sure not out of bounds. If not, set rightVal
        if currCol - 1 > -1:
            leftVal = Eh[currRow - 1][currCol - 1]

        # Check to make sure not out of bounds. If not, set leftval
        if currCol + 1 < overlap_distance:
            rightVal = Eh[currRow - 1][currCol + 1]

        # Curr col is set to lowest value's col, curr row is decremented

        if leftVal < rightVal and leftVal < middleVal:
            currCol = currCol - 1
        elif rightVal < leftVal and rightVal < middleVal:
            currCol = currCol + 1

        currRow = currRow - 1

    return (mask * oldpatch) + ((1 - mask) * inpatch)

"""
    Helper function that will compute the sum of squared error
    for two given input vectors
"""
def sse_error(v1, v2):
    return math.pow(l2_norm(v1) - l2_norm(v2), 2)

def construct_new_patch():
    print("Implement me")

def get_all_patches(Io, patchsize):
    patches = []

    # Get the current dimensions
    imwidth, imheight = Io.shape[:2]

    for row in range(0, imheight - patchsize, patchsize):
        for col in range(0, imwidth - patchsize, patchsize):
            patches.append((row, col))

def main():
    image_quilting("input/wrinkly_boy.jpg")

main()