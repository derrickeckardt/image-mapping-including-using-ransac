# CS B657 - Assignment 2: Warping, Matching, Stitching, Blending

Completed by Derrick Eckardt on March 19, 2019.  Please direct any questions to [derrick@iu.edu](mailto:derrick@iu.edu)

The assignment prompt can be found at [Assignment 2 Prompt](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/a2-sp2019.pdf)

# General Comments
I think the coolest thing 


# Part 1 - Clustering

## How many matches, first or second best, and not using Non Maximal Suppression
The first big design decision I came across was how many matches. OpenCV's detectAndCompute uses 500 by default, and the samples showed a 1000.  To find 1000 images and then perform matching on them, it took close to an hours to run the program.  That wasn't practical.

The other part of that initial decision is what should constitute a match?

So that made me want to use Non-maximal suppression(NMS), to eliminate duplicate points and some of the garbage points.  It usually reduced matches by a factor of two.  However, when I ran NMS, it eliminated points, but it didn't really get rid of the garbage points.  So, it helped get runtime down to under an hour, however, the results were just as bad.

Next item I looked at was whether I should use the first best match or the second best match. I ran sample tests to see what my images looked like with, with eiffel18 and iffel 19, since they are remarkably similar images.  Here are the results for best match:

![eiffel comparison test first best match](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_eiffel_best_match_no_nms.jpg)

and for second best match:

![eiffel comparison test second best match](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_eiffel_second_match_no_nms.jpg)

As you can see, the second best match gave no better matches, and actually got rid of the some of the good ones.

When all else fails, read the documentation!  In the CV2 documentation, I saw that it ranks the matches and provides the strongest candidates.  Meaning, the more points you ask for, the worse they likely are.

So, I dropped to just 100 features, and I actually got some really interesting results.  First, look at this comparison of two colosseum images:

![colosseum few matches](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_colosseum_best_match.jpg)

There were only about 8 matching lines, but they are dead on.  Then, when I looked at Big Ben, I saw a similar response:

![big ben few matches](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_bigben_best_match_no_nms.jpg)

It matched the faces of the clock.  At this point, I decided I would actually focus on few points, and use those for determing what my matches would be.

## Implementing RANSAC - Stealing from the Future
After I did Part 3, I realized that I could use my RANSAC algorithm from Part 3, to clean-up my matches in Part 1.  So, skip ahead to Part 3 if you want to know how I implemented RANSAC.  Here, you can see see the results of my eiffel tower matches, starting with 200 feaures, no NMS, and then without RANSAC implemented:

![eiffel without RANSAC](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_eiffel.jpg)

And then with RANSAC implemented.

![eiffel with RANSAC](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_ransac_eiffel.jpg)

All the bad matches are gone. I can only visually see one outlier there.  It even successfully matches the streetlights to the bottom left of the Eiffel tower.





## How 


## How should I value connections?
Initially, I went with a very dumb heurestic to see how two images were related, and that was the number of connections.  In fact, you'll see my dictionary of edge weights referred to as "common\_points\_matrix".  It was a terribly metric, but was good for doing quick testing to see if my algorithm would work.  The results were not any better than randomly placing them into different centroids.  In fact, it wasn't even that good at times, because it would tend to almost all of the images in one clusters, and then leave the other clusters with just two or three images, and sometimes just the centroid.

After I implemented my decisions to use fewer features, not use Non-o that I'm more likely to get higher quality features, 



## How should I cluster?


## Results

## Recommendations for Improving Part 1

# Part 2 - Transformation

## What size should my output image be?
This was something that early on, I thought would be straightforward, which like most items here, turned out to have some intricarcies.  Depending on the transform matrix applied, particularly for Rigid or Affine transformations, I would see my image get cut-off on the screen.

I attempted to resolve this with a function called get_shape(), which looked at the corners of the base image, forward warped them to determine what the bounding box would be, along with any x,y offsets to account for the fact that the warping could mean that it wanted a point from a negative spot.  This would also minimize the amount of white space.

However, when I implemented get_shape, what was lost was the fact that in comparison to the original image, it was no longer always obvious that something had happened in every dimension.  For example, the Rigid transformations now just looked like rotations, with the 

Ultiamtely, as a design decision, I decided to go with the original size of the image for cases, 2,3, and 4, so you could best see the difference in the transformation.  For case 1, I changed the shape so that the translated portion would appear black.  I left the function and offsets in place to improve on this in the future.

## Results

## Recommendations for Improving Part 2
Here are some thoughts on how I could improve this part of my program.

### Better Visualize the Movement
As I stated in the "what size should my output image be?" section, I think there is a better way to visualize the transformations, than I have implemented.  If I had another cut at it, or use for this program, I would work further on this seemingly easy problem, that is not so easy.  

# Part 3 - Merging

## RANSAC

## Results

## Recommendations for Improving Part 3

