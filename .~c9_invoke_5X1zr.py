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
# - create better boudning boxes for part 2, 3, and perhaps 4
# - Speed up bilineal by using a dictionary?  (Tried it, was acutally slower)
# - refactor code to reduce redundancies between each
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
from math import atan, cos, sin
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

def make_im_dict(im):
    height,width,depth = im.shape
    im_dict ={}
    for j in range(height):
        im_dict[j]={}
        for i in range(width):
            # print(j,i)
            im_dict[j][i] = im[j,i]
    return im_dict

def merge(im1,im2):
    pass


# does bilineal interpopulation for inverse warping
# return the pixel value
def bilineal(im,x,y):
    dx = x-int(x)
    dy = y-int(y)
    spacex = 1 if dx != 0 else 0
    spacey = 1 if dy != 0 else 0
    pixel = im[int(y-dy),int(x-dx)] * (1-dx)*(1-dy) + \
        im[int(y)+spacey,int(x)+spacex] * dx * dy + \
        im[int(y-dy),int(x)+spacex] * dx * (1-dy) + \
        im[int(y)+spacey,int(x-dx)] * (1-dx) * dy
    return pixel

    # bilinial(warp_im,230.01,500.01)       
    # # print(230.01,500.01)
    # bilinial(warp_im,230.99,500.01)       
    # # print(230.99,500.01)
    # bilinial(warp_im,230.01,500.99)       
    # # print(230.01,500.99)
    # bilinial(warp_im,230.99,500.99)       
    # # print(230.99,500.99)

def inversewarp(base_im,transform_matrix_inv, output_shape):
    height, width, depth = output_shape
    base_height, base_width, base_depth = base_im.shape
    output_im = np.zeros((height,width,depth), np.uint8)  # https://stackoverflow.com/questions/12881926/create-a-new-rgb-opencv-image-using-python

    for x in range(width):
        for y in range(height):
            pixel = np.array([x, y, 1])
            xyw = np.matmul(transform_matrix_inv,pixel)
            x_o, y_o = xyw[0] / xyw[2], xyw[1] / xyw[2]
            if x_o >= 0 and x_o <= base_width-1 and y_o >= 0 and y_o <= base_height-1:
                # print(x_o,y_o)
                output_im[y,x] = bilineal(base_im,x_o,y_o)
            # else:
                # no change, essentially output_im[y,x] = [0 0 0]

    return output_im

# takes as input of an image as described by keypoints and descriptors that have
# been placed in a dictorary, in the form:
    # image[i] = {'keypoints': keypoints[i], 'descriptors':descriptors[i]}
    # image["name"] = image (as string)
# where keypoints, descriptors are generated from:
    # img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # orb = cv2.ORB_create(nfeatures=1000)
    # keypoints, descriptors = orb.detectAndCompute(img, None)
# and then saved in as a dictionary via:
#     for i in range(len(keypoints)):
#             image[i] = {'keypoints':keypoints[i], 'descriptors':descriptors[i]}
def match_images(image1, image2):
    k = 0
    matches = {}
    if image1['name'] != image2['name']:# and image1 == 'part1-images-small/bigben_6.jpg':  # and image2 == 'part1-images-small/bigben_2.jpg'
        for i in image1['orb']:
            # initial top two matches.
            best_matches = {1:{"distance":float('inf'),"keypoint1":image1['orb'][i]['keypoints'],"keypoint2":image1['orb'][i]['keypoints']},
                            2:{"distance":float('inf'),"keypoint1":image1['orb'][i]['keypoints'],"keypoint2":image1['orb'][i]['keypoints']}}
            for j in image2['orb']:
                distance = cv2.norm(image1['orb'][i]['descriptors'], image2['orb'][j]['descriptors'], cv2.NORM_HAMMING)
                if distance <= 50:
                    if distance <= best_matches[1]['distance']:# this threshold impacts speed...
                        best_matches[2] = best_matches[1]
                        best_matches[1] = {"distance":distance,"keypoint1":image1['orb'][i]['keypoints'],"keypoint2":image2['orb'][j]['keypoints']}
                    elif distance <= best_matches[2]['distance']:
                        best_matches[2] = {"distance":distance,"keypoint1":image1['orb'][i]['keypoints'],"keypoint2":image2['orb'][j]['keypoints']}
            # if best_matches[1]["distance"] != float('inf') and best_matches[2]['distance'] != float('inf'):
                # print(best_matches[1]['distance'],best_matches[2]['distance'] )
            if best_matches[1]['distance'] != float('inf'):
                matches[k] = best_matches[1]
                k += 1
    print(k, len(matches)   ) 
    return k, matches
    
def simple_refs(refs):
    x, y,xp, yp = {}, {}, {}, {}
    i = 1
    # for readability later
    for each in refs:
        [x[i],y[i]],[xp[i],yp[i]] = each
        i += 1
    return x, y, xp, yp

def four_point_tranform_matrix(x,y,xp,yp):
    pointmatrix = np.array([[x[1],y[1],1,0,0,0,-x[1]*xp[1],-y[1]*xp[1]],
                        [0,0,0,x[1],y[1],1,-x[1]*yp[1],-y[1]*yp[1]],
                        [x[2],y[2],1,0,0,0,-x[2]*xp[2],-y[2]*xp[2]],
                        [0,0,0,x[2],y[2],1,-x[2]*yp[2],-y[2]*yp[2]],
                        [x[3],y[3],1,0,0,0,-x[3]*xp[3],-y[3]*xp[3]],
                        [0,0,0,x[3],y[3],1,-x[3]*yp[3],-y[3]*yp[3]],
                        [x[4],y[4],1,0,0,0,-x[4]*xp[4],-y[4]*xp[4]],
                        [0,0,0,x[4],y[4],1,-x[4]*yp[4],-y[4]*yp[4]]
                        ])

    pointmatrix_inv = np.linalg.inv(pointmatrix)
    primematrix = np.array([xp[1],yp[1],xp[2],yp[2],xp[3],yp[3],xp[4],yp[4]])

    # or could have used print(np.linalg.solve(pointmatrix,primematrix))
    tmatrix = np.matmul(pointmatrix_inv,primematrix)
    tmatrix = np.reshape(np.append(tmatrix,[1]),(3,3))

    return tmatrix

def part1():
    starttime = time.time()
    k, output_file = int(sys.argv[2]),sys.argv[-1]
    input_images = {}
    for i in range(3,len(sys.argv)-1):
        input_images[i-3] = sys.argv[i]

    # get keypoints and descriptors for each image
    print("Beginning loading images,  Expected run time for this part is approximately "+str(round(len(input_images)/60,3))+" seconds.")
    orb_images = {}
    for key, image in input_images.items():
        orb_images[image] = {"name":image, "orb":{}}
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        for i in range(len(keypoints)):
            orb_images[image]['orb'][i] = {'keypoints':keypoints[i], 'descriptors':descriptors[i]}
        
        # non-maximum suppression
        nonmax, i = False, 0
        while nonmax == False:
            for j in range(i,len(orb_images[image]['orb'])):
                if within_circle(orb_images[image]['orb'][i]['keypoints'],orb_images[image]['orb'][j]['keypoints']):
                    if orb_images[image]['orb'][i]['keypoints'].response > orb_images[image]['orb'][j]['keypoints'].response:
                        # remove j item
                        orb_images[image]['orb'].pop(j)
                    else:
                        #remove i item and break loop
                        orb_images[image]['orb'].pop(i)
                        break
            i += 1
            if i > len(orb_images[image]['orb']):
                nonmax = True
    print("Completed loading all "+str(len(orb_images)) +" images with Non Maximal Suppression in "+str(round(time.time() - starttime,3))+" seconds.")

    # match the points
    print("Beginning image matching.  Expected run time for this part is appoximately "+str(round((len(orb_images)*(len(orb_images)-1))/3,3))+" seconds.")
    starttime = time.time()
    common_points_matrix = {}
    for image1 in orb_images.keys():
        common_points_matrix[image1] = {}
        for image2 in orb_images.keys():
            common_points_matrix[image1][image2] = match_images(orb_images[image1],orb_images[image2])
    print("Completed matching all "+str(len(orb_images)*(len(orb_images)-1)) +" pairs of images in "+str(round(time.time() - starttime,3))+" seconds.")

    # kMeans grouping -- I select k random images as the centroids, then decide 
    # which group the remaining images goes to based on which one it has more
    # connections with -- huge opportunity for improvement
    print("Beginning clustering via k-Means.  Estimated run time is a blink of the eye.")
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
    # print(common_points_matrix["part1-images/eiffel_18.jpg"]["part1-images/eiffel_19.jpg"])
    # print(common_points_matrix)
    # pprint(common_points_matrix)
    print("Completed kmeans clustering of "+str(len(orb_images))+" into "+str(k)+" clusters in "+str(round(time.time() - starttime,3))+" seconds.")
    
    # Insert clustering output
    
    print("Clusters can be viewed in '"+output_file+"'.  I hope you like my clusters!")

def part2():
    starttime = time.time()
    n = int(sys.argv[2])
    base_im_file, warp_im_file, output_im_file = sys.argv[3:6]
    refs = [[[int(j) for j in sys.argv[6+2*i].split(",")], [int(k) for k in sys.argv[7+2*i].split(",")]] for i in range(n)]
    x, y, xp, yp = simple_refs(refs)

    # # Test Matrix for Lincoln Test Case
    # tmatrix = np.array([[0.907, 0.258, -182],
    #                     [-0.153, 1.44, 58],
    #                     [-0.000306, 0.000731, 1]])
    # tmatrix_inv = np.linalg.inv(tmatrix)
    # test_im = cv2.imread("part2-images/lincoln.jpg")
    # output_im = inversewarp(test_im,tmatrix_inv)
    # cv2.imwrite("part2-images/lincoln_test.jpg",output_im)

    # Load and create images
    base_im = cv2.imread(base_im_file)
    warp_im = cv2.imread(warp_im_file)
    
    if n == 1:      # translation
        dx, dy = xp[1] - x[1], yp[1] - y[1]
        # Find translation matrix
        translation_matrix = np.array([[1,0,dx],
                                       [0,1,dy],
                                       [0,0,1]
                                       ])
        translation_matrix_inv = np.linalg.inv(translation_matrix)
        
        # Create shape for output image
        translation_shape = (base_im.shape[0] + dy, base_im.shape[1] + dx, base_im.shape[2])
        
        # create output image
        output_im = inversewarp(base_im,translation_matrix_inv, translation_shape)
        cv2.imwrite(output_im_file, output_im)
        print("Image '"+base_im_file+"' has been translated and saved as '"+output_im_file+"'.  That was a nice move." )

    elif n == 2: # Euclidean (rigid)
        theta1 = atan((y[2]-y[1]) / (x[2]-x[1]))
        theta2 = atan((yp[2]-yp[1]) / (xp[2]-xp[1]))
        theta = theta2-theta1  # radians

        dx, dy = xp[1] - x[1], yp[1] - y[1]
        # Find translatiom matrix
        translation_matrix = np.array([[cos(theta),-sin(theta),dx],
                                       [sin(theta),cos(theta),dy],
                                       [0,0,1]
                                       ])
        translation_matrix_inv = np.linalg.inv(translation_matrix)
 
         # Create shape for output image
        # translation_shape = (base_im.shape[0] + dy, base_im.shape[1] + dx, base_im.shape[2])
        
        # create output image
        output_im = inversewarp(base_im,translation_matrix_inv, base_im.shape)  #translation_shape
        cv2.imwrite(output_im_file, output_im)
        print("Image '"+base_im_file+"' has made a rigid transformation and is saved at '"+output_im_file+"'.  Nice spinning, DJ." )
       
    elif n == 3:    # Affine
        # calculate affine matrix, and then it's reverse
        pointmatrix = np.array([[x[1],y[1],1,0,0,0],
                                [0,0,0,x[1],y[1],1],
                                [x[2],y[2],1,0,0,0],
                                [0,0,0,x[2],y[2],1],
                                [x[3],y[3],1,0,0,0],
                                [0,0,0,x[3],y[3],1]
                                ])
        pointmatrix_inv = np.linalg.inv(pointmatrix)
        primematrix = np.array([xp[1],yp[1],xp[2],yp[2],xp[3],yp[3]])

        amatrix = np.matmul(pointmatrix_inv,primematrix)
        print(amatrix)
        amatrix = np.reshape(np.append(amatrix,np.array([0, 0, 1])),(3,3))
        print(amatrix)

        # take the inverse of the affine matrix
        amatrix_inv = np.linalg.inv(amatrix)
        
        output_im = inversewarp(base_im,amatrix_inv, base_im.shape)
        cv2.imwrite(output_im_file, output_im)
        print("Image '"+base_im_file+"' has made an affice transformation and saved as '"+output_im_file+"'.  That was tripy" )

    elif n == 4:    # Projective
        # calculate transform matrix, and then its inverse
        tmatrix = four_point_tranform_matrix(x,y,xp,yp)
        tmatrix_inv = np.linalg.inv(tmatrix)

        output_im = inversewarp(base_im,tmatrix_inv, base_im.shape)
        cv2.imwrite(output_im_file, output_im)
        print("Image '"+base_im_file+"' has been translated and saved as '"+output_im_file+"'.  I feel bent out of shape." )


    else:
        print("You have entered a tranform this program cannot make.  Please check your inputs.)")

def part3():
    # get command line inputs
    im1_file, im2_file, output_im_file = sys.argv[2:5]

    # Load and create images, Find interest points for images, load keypoint descriptors
    img1 = cv2.imread(im1_file, cv2.IMREAD_GRAYSCALE)
    orb1 = cv2.ORB_create(nfeatures=200)
    keypoints1, descriptors1 = orb1.detectAndCompute(img1, None)
    im1 = {"name":im1_file, "orb":{}}
    for i in range(len(keypoints1)):
        im1['orb'][i] = {'keypoints':keypoints1[i], 'descriptors':descriptors1[i]}

    img2 = cv2.imread(im2_file, cv2.IMREAD_GRAYSCALE)
    orb2 = cv2.ORB_create(nfeatures=200)
    keypoints2, descriptors2 = orb2.detectAndCompute(img2, None)
    im2 = {"name":im2_file, "orb":{}}
    for i in range(len(keypoints2)):
        im2['orb'][i] = {'keypoints':keypoints2[i], 'descriptors':descriptors2[i]}
    
    # Perform matching on them
    common_points, matches = match_images(im1,im2)
    
    
    matches_list = []
    for m in range(common_points):
        # print(matches[m]['keypoint1'].pt)
        # print(matches[m]['keypoint2'].pt)
        # keypoints1.extend(matches[m]['keypoint1'])
        # keypoints2.extend(matches[m]['keypoint2'])
        matches_list.append([matches[m]['keypoint1'], matches[m]['keypoint2']]) 

    # display for testing
    # based on testing code provide to me by Priyank Sharma, simplified and
    # modified to work with my variable environment and datastructures
    match_im = np.hstack((img1,img2))
    
    for p1,p2 in matches_list:
        from_cord = (int(p1.pt[0]),int(p1.pt[1]))
        to_cord = (img1.shape[1]+int(p2.pt[0]),int(p2.pt[1]))
        cv2.line(match_im,from_cord,to_cord,(255,0,0),1)

    cv2.imwrite("matching_test.jpg", match_im)
    

    # match_im = cv2.drawMatches(img1,keypoints1, img2,keypoints2, matches_list, None, flags=2)
    # cv2.imwrite("matching_test.jpg",match_im)

    
    # Perform Ransac on the images
    

    #Placeholder refs until I get RANSAC done
    refs = [[[141,131],[318,256]],[[480,159],[534,372]],[[493,630],[316,670]],[[64,601],[73,473]]]

    # Calculate the transformation matrix from four points from ransac
    x, y, xp, yp = simple_refs(refs)
    
    # transform one image to look the other
    # calculate transform matrix, and then its inverse
    tmatrix = four_point_tranform_matrix(x,y,xp,yp)
    tmatrix_inv = np.linalg.inv(tmatrix)
    
    #reload images as color images now
    img1 = cv2.imread(im1_file)
    img2 = cv2.imread(im2_file)

    # project one image onto another one
    image1_on_2 = inversewarp(img1,tmatrix_inv, img1.shape)

    
    # merge the two images
    height,width,depth = image1_on_2.shape
    output_im = np.zeros((height,width,depth), np.uint8) 
    for y in range(height):
        for x in range(width):
            # output_im[y,x] = np.divide(np.add(img2[y,x],image1_on_2[y,x]),2.0)
            r = (float(img2[y,x][0]) + float(image1_on_2[y,x][0]))/2.0
            g = (float(img2[y,x][1]) + float(image1_on_2[y,x][1]))/2.0
            b = (float(img2[y,x][2]) + float(image1_on_2[y,x][2]))/2.0

            output_im[y,x] = np.array([r, g,b])

    cv2.imwrite(output_im_file, output_im)
    
if part == "part1":
    part1()
    # profile.run("part1()")
    # cProfile.run("part1()")
elif part == "part2":
    part2()
    # cProfile.run("part2()")
    # profile.run("part2()")
elif part == "part3":
    part3()
    # cProfile.run("part2()")
else:
    print("You did not enter 'part1', 'part2', or 'part3' to run the program. Please check your input.  Thank you.")

