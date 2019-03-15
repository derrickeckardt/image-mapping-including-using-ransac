#!/usr/bin/env python3
#
#   ./a2.py : Grading automated scanned answer sheet:
#
# For Part 1,
#   ./a2.py part1 k img_1.png img_2.png ... img_n.png output_file.txt
#
# For Part 2
#   ./a2.py part2 n img_1.png img_2.png img_output.png img1_x1,img1_y1 img2_x1,img2_x1 ... img1_xn,img1_yn img2_xn,img2_yn
#
# For Part 3
#   ./a2.py part3 image_1.jpg image_2.jpg output.jpg
# 
################################################################################
# CS B657 Spring 2019, Assignment #2 -B657 Assignment 2: Warping, Matching, Stitching, Blending
#
# Completed by:
# Derrick Eckardt
# derrick@iu.edu
# 
# Completed on March 19, 2019. For the assignment details, please visit:
#
#   https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/a2-sp2019.pdf
#
# For a complete details on this program, please visit the readme.md file at:
#
#   https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/readme.md
#
################################################################################

################################################################################
# To fix Part 1
# - match to second best match
# - record the points that are the match
# - fix the inclusion of the directory where the images are, or ask Piazza Q
# - adjust strength of link based on how good a match
# - add voting somehow for matches
# - Take the best x number (50?) best matches
# - Add file output
# - Add dynamic programming to speed up?
# - For write-up -- how do we speed it up?
# - functionize part 1 into parts
#
# To fix Part 2
# 
# - Speed up bilineal by using a dictionary?
#
#
################################################################################


# import libraries
import sys
import cProfile
import numpy as np
import scipy
import time
import cv2
import random
import imageio
from pprint import pprint
import profile

part = sys.argv[1]

# Checks to see if a keypoint is within another keypoint
def within_circle(keypoint1, keypoint2):
    proj = (keypoint1.size/2) / (2 ** (0.5))
    if keypoint1.pt[0] - proj < keypoint2.pt[0] and keypoint1.pt[0] + proj > keypoint2.pt[0]:
        if keypoint1.pt[1] - proj < keypoint2.pt[1] and keypoint1.pt[1] + proj > keypoint2.pt[1]:
            return True
        else:
            return False
    else:
        return False

def part1():
    starttime = time.time()
    k, output_file = int(sys.argv[2]),sys.argv[-1]
    input_images = {}
    for i in range(3,len(sys.argv)-1):
        input_images[i-3] = sys.argv[i]

    # get keypoints and descriptors for each image
    orb_images = {}
    for key, image in input_images.items():
        orb_images[image] = {}
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        for i in range(len(keypoints)):
            orb_images[image][i] = {'keypoints':keypoints[i], 'descriptors':descriptors[i]}
        
        
        # non-maximum suppression
        nonmax, i = False, 0
        while nonmax == False:
            for j in range(i,len(orb_images[image])):
                if within_circle(orb_images[image][i]['keypoints'],orb_images[image][j]['keypoints']):
                    if orb_images[image][i]['keypoints'].response > orb_images[image][j]['keypoints'].response:
                        # remove j item
                        orb_images[image].pop(j)
                    else:
                        #remove i item and break loop
                        orb_images[image].pop(i)
                        break
            i += 1
            if i > len(orb_images[image]):
                nonmax = True
    print("load all images ", time.time() - starttime)

    # match the points
    starttime = time.time()
    common_points_matrix = {}
    for key1, image1 in input_images.items():
        common_points_matrix[image1] = {}
        for key2, image2 in input_images.items():
            common_points = 0
            if image1 != image2:# and image1 == 'part1-images-small/bigben_6.jpg':  # and image2 == 'part1-images-small/bigben_2.jpg'
                for i in orb_images[image1]:
                    for j in orb_images[image2]:
                        distance = cv2.norm(orb_images[image1][i]['descriptors'], orb_images[image2][j]['descriptors'], cv2.NORM_HAMMING)
                        if distance <= 50:  # this threshold impacts speed...
                            common_points += 1
                            break
            common_points_matrix[image1][image2] = common_points
    #         print(image1, image2, common_points, time.time() - starttime)
            
    # print("my bfer", pprint(common_points_matrix))
    print("Descriptor matching", time.time() - starttime)

    # kMeans grouping
    # we select k random images as the centroids, then we decide which group it goes to
    # based on which one it has more connections with
    starttime = time.time()
    all_images = list(input_images.values())
    random.shuffle(all_images)
    centroids = all_images[0:k]

    for z in range(100):
        print("Run",z,"Centroids:", sorted(centroids))
        groupings = {}
        for centroid in centroids:
            groupings[centroid] = []
        for image in list(input_images.values()):
            max_edge = [0, ""]  # may need to reverse this if I do it by strength of match
            for centroid in centroids:
                if image not in centroids:
                    edge_score = common_points_matrix[centroid][image] + common_points_matrix[image][centroid]
                    if edge_score >= max_edge[0]:
                        max_edge = [edge_score, image, centroid]
            if image not in centroids:
                groupings[max_edge[2]].extend([max_edge[1]])
        # pprint(groupings)
        old_centroids = centroids*1
        centroids = []
        for centroid, images in groupings.items():
            group_images = images + [centroid]
            max_node = [0,""]  # may need to reverse this if I do it by strength of match
            for image1 in group_images:
                node_score = sum([common_points_matrix[image1][image2] + common_points_matrix[image2][image1] for image2 in group_images])
                if node_score >= max_node[0]:
                    max_node = [node_score,image1]
            centroids.extend([max_node[1]])
        if sorted(old_centroids) == sorted(centroids):
            break
    
    
    print("Final Centroids:", sorted(centroids))
    pprint(groupings)
    pprint(common_points_matrix)
        # 
                
                

    
    print("kmeans matching", time.time() - starttime)

def make_im_dict(im):
    height,width,depth = im.shape
    im_dict ={}
    for j in range(height):
        im_dict[j]={}
        for i in range(width):
            # print(j,i)
            im_dict[j][i] = im[j,i]
    return im_dict

# does bilineal interpopulation for inverse warping
# return the pixel value
def bilineal(im,x,y):
    dx = x-int(x)
    dy = y-int(y)
    pixel = im[int(y-dy),int(x-dx)] * (1-dx)*(1-dy) + \
        im[int(y)+1,int(x)+1] * dx * dy + \
        im[int(y-dy),int(x)+1] * dx * (1-dy) + \
        im[int(y)+1,int(x-dx)] * (1-dx) * dy
    return pixel

    # bilinial(warp_im,230.01,500.01)       
    # # print(230.01,500.01)
    # bilinial(warp_im,230.99,500.01)       
    # # print(230.99,500.01)
    # bilinial(warp_im,230.01,500.99)       
    # # print(230.01,500.99)
    # bilinial(warp_im,230.99,500.99)       
    # # print(230.99,500.99)

def part2():
    starttime = time.time()
    n = int(sys.argv[2])
    base_im_file, warp_im_file, output_im_file = sys.argv[3:6]
    refpoints = [[[int(j) for j in sys.argv[6+2*i].split(",")], [int(k) for k in sys.argv[7+2*i].split(",")]] for i in range(n)]
    # pprint(refpoints)

    # Test Matrix for Lincoln Test Case
    tmatrix = np.array([[0.907, 0.258, -182],
                        [-0.153, 1.44, 58],
                        [-0.000306, 0.000731, 1]])
    tmatrix_inv = np.linalg.inv(tmatrix)
    test_im = cv2.imread("part2-images/lincoln.jpg")
    output_im = inversewarp(test_im,tmatrix_inv)
    cv2.imwrite("part2-images/lincoln_test.jpg",output_im)

    # Load and create images
    base_im = cv2.imread(base_im_file)
    warp_im = cv2.imread(warp_im_file)
    
    if n == 1:      # translation
        pass
    elif n == 2:    # Euclidean (rigid)
        pass
    elif n == 3:    # Affine
        pass
    elif n == 4:    # Projective
        # calculate transform matrix, and then its inverse
        tmatrix = tmatrix

        # take the inverse of the transform matrix
        tmatrix_inv = np.linalg.inv(tmatrix)

        output_im = inversewarp(base_im,tmatrix_inv)
        cv2.imwrite(output_im_file, output_im)

    else:
        print("You have entered a tranform this program cannot make.  Please check your inputs.)")

def inversewarp(base_im,transform_matrix_inv):
    height, width, depth = base_im.shape
    output_im = np.zeros((height,width,depth), np.uint8)  # https://stackoverflow.com/questions/12881926/create-a-new-rgb-opencv-image-using-python

    for x in range(width):
        for y in range(height):
            pixel = np.array([x, y, 1])
            xyw = np.matmul(transform_matrix_inv,pixel)
            x_o, y_o = xyw[0] / xyw[2], xyw[1] / xyw[2]
            if x_o >= 0 and x_o < width-1 and y_o >= 0 and y_o < height-1:
                output_im[y,x] = bilineal(base_im,x_o,y_o)
            # else:
                # no change, essentially output_im[y,x] = [0 0 0]

    return output_im
    
    
        




if part == "part1":
    #part1()
    # profile.run("part1()")
    cProfile.run("part1()")
elif part == "part2":
    part2()
    # cProfile.run("part2()")
    # profile.run("part2()")
elif part == "part3":
    print(part)
else:
    print("You did not enter 'part1', 'part2', or 'part3' to run the program. Please check your input.  Thank you.")
