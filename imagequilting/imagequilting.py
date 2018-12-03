import cv2 as cv
import math
import random
import numpy

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
    Is = init_is(Io, (resized_width, resized_height, imdim), patchsize)

    patch = select_compatible_patch(Io, get_random_patch(Io, patchsize), patchsize, tolerance, 0)

    #for row in range(0, imsize[0]):
    #    for col in range(0, imsize[1]):
    #        patch = Is[row][col]

    # Write result to output directory
    #outfile = "output/" + inputfile
    #cv.imwrite("output/brick.jpg", Is)

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
def init_is(Io, resized_dim, patchsize):
    # Initialize new image
    Is = numpy.zeros(resized_dim)

    # Generate a random patch
    rand_patch = get_random_patch(Io, patchsize)

    # Copy over patch information into 0, 0
    for row in range(0, patchsize):
        for col in range(0, patchsize):

            # Copy over patch
            Is[row][col] = rand_patch[row][col]

    return Is


'''def randomized_quilt(Io, resized_dim, num_patches, patchsize, overlap_distance):
    # Initialize new image
    Is = numpy.zeros(resized_dim)

    # Variables that will keep track of where we are starting the copy in Is
    row_offset = col_offset = 0

    # For all the patches that should be in the new image:
    # 1. Generate a new random patch
    # 2. Copy that random patch over to the new image
    for p_row in range(0, num_patches[0]):
        for p_col in range(0, num_patches[1]):
            # Generate a new random patch
            rand_patch = get_random_patch(Io, patchsize)

            # Copy over patch information, overlapping with previous layer
            for row in range(0, patchsize):
                for col in range(0, patchsize):

                    # Skip if copying over this part of the patch will be out of bounds
                    if row + row_offset >= resized_dim[0] or col + col_offset >= resized_dim[1]:
                        continue

                    # Copy over patch
                    Is[row + row_offset][col + col_offset] = rand_patch[row][col]

            # Update the column offset as you go across the columns so it overlaps
            col_offset += (patchsize - overlap_distance)

        # Update col_offset and row_offset as you start a new row
        col_offset = 0
        row_offset += (patchsize - overlap_distance)

    return Is'''


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

    # initialize D-image (distance image)
    d_rows = imwidth - patchsize
    d_cols = imheight - patchsize
    D = numpy.zeros((d_rows, d_cols))

    print("Calculating D image")
    # For every pixel m, n in range (0...M - patchsize, 0...N - patchsize),
    # Imaging (m, n) is the top left corner of a patch. Compute the distance measure
    # for that patch and save it to the D image
    for row in range(0, d_rows):
        for col in range(0, d_cols):
            D[row][col] = compute_D(Io, (row, col), oldpatch, patchsize, overlap_distance)

    print("Finding dmin")
    # Find the minimum distance. Will be used with tolerance to filter out patches
    d_min = (0, 0)
    for row in range(0, d_rows):
        for col in range(0, d_cols):
            if D[d_min[0]][d_min[1]] > D[row][col]:
                d_min = (row, col)

    # Calculate minimum value
    minimum_val = (1 + tolerance) * D[d_min[0]][d_min[1]]

    print("Getting all viable patches")
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
def compute_D(Io, coords, oldpatch, patchsize, overlap_distance):
    m, n = coords[0], coords[1]
    distance = 0

    for i in range(0, patchsize):
        for j in range(0, patchsize):
            # Q(i, j) is 0 if not in overlap region, 1 otherwise
            # So if we aren't in the overlap region, don't bother computing anything
            if patchsize - overlap_distance < i and patchsize - overlap_distance < j:
                continue

            distance += l2_norm(numpy.square(numpy.subtract(oldpatch[i][j], Io[m + i][n + j])))

    return distance

"""
    Helper function that computes the L2 Norm of 
    a vector value
"""
def l2_norm(v1):
    squared_sum = 0
    for i in range(0, v1.shape[0]):
        squared_sum += math.pow(v1[i], 2)
    return math.sqrt(squared_sum)


def compute_min_boundary_cut():
    print("Implement me")


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