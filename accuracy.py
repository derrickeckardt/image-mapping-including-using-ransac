#!/usr/bin/env python3
from pprint import pprint;
import sys
from collections import Counter

output_file = sys.argv[1]

groups = []
i = 0
with open(output_file, "r") as file:
    for line in file:
        groups.append([file.split("_")[0] for file in line.split()])
        
all_images = Counter()
for group in groups:
    all_images += Counter(group)

total_images = len(list(all_images.elements()))

TP = 0
TN = 0
for group in groups:
    image_count = Counter(group)
    for key, value in image_count.items():
        TP += value * (value - 1)
        TN += value* (total_images - all_images[key] - (len(group) - value))

print(round((TP+TN)/(0.01*total_images*(total_images-1)),2))