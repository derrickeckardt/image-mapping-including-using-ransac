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

# import libraries
import sys
import cProfile

part = sys.argv[1]

def part1():
    k, output_file = sys.argv[2],sys.argv[-1]
    input_images = {}
    for i in range(3,len(sys.argv)-1):
        input_images[i-3] = sys.argv[i]
    print(len(sys.argv))
    print(len(input_images))
    print(input_images)
    print(part)

if part == "part1":
    cProfile.run("part1()")
elif part == "part2":
    print(part)
elif part == "part3":
    print(part)
else:
    print("You did not enter 'part1', 'part2', or 'part3' to run the program. Please check your input.  Thank you.")
