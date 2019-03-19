# CS B657 - Assignment 2: Warping, Matching, Stitching, Blending

Completed by Derrick Eckardt on March 19, 2019.  Please direct any questions to [derrick@iu.edu](mailto:derrick@iu.edu)

The assignment prompt can be found at [Assignment 2 Prompt](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/a2-sp2019.pdf)

# General Comments
I think the coolest thing about this assignment, is that it becomes really easy to understand how our a lot of our every day technology actually works.  By the end, I've taken two different images of the same thing, summed them together, and got some startinly cool results!  Most interesting, was the cyclical nature of this assignment.  After I did part 1, and part 2, I was able to use things I learned or coded in part 3 to go back and improve part 1 and part 2. I have a lot else to say on specific points, so let's get on with the show.

# Part 1 - Clustering
Wow. There were so many tiny decisions to make.  The biggest thing is having to deciding what to optimize for.  There was way more 'expert' knowledge that was needed in order to provide decent results.

## How many matches, first or second best, and not using Non Maximal Suppression
The first question I had what s what should constitute a match?  I found the hamming distance between two points.  In theory, a match would be when the distance is zero.  Practically speaken, that almost never happened. Ultimately, I played around with a threshold of a normalized distance less than 75.  That was enough to allow close matches in, but keep up the really absurd matches.

The second, and bigger design decision I came across was how many matches. OpenCV's detectAndCompute uses 500 by default, and the samples showed a 1000.  To find 1000 images and then perform matching on them, it took close to an hours to run the program.  That wasn't practical.

So that made me want to use Non-maximal suppression(NMS), to eliminate duplicate points and some of the garbage points.  It usually reduced matches by a factor of two.  However, when I ran NMS, it eliminated points, but it didn't really get rid of the garbage points.  So, it helped get runtime down to under an hour, however, the results were just as bad.  You can see my implementation of NMS in the code, commented out.

Next item I looked at was whether I should use the first best match or the second best match. I ran sample tests to see what my images looked like with, with eiffel18 and iffel 19, since they are remarkably similar images.  Here are the results for best match:

![eiffel comparison test first best match](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_eiffel_best_match_no_nms.jpg)

and for second best match:

![eiffel comparison test second best match](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_eiffel_second_match_no_nms.jpg)

As you can see, the second best match gave no better matches, and actually got rid of the some of the good ones.

When all else fails, read the documentation!  In the CV2 documentation, I saw that it ranks the matches and provides the strongest candidates.  Meaning, the more points you ask for, the worse they likely are.  So, I dropped to just 100 features, and I actually got some really interesting results.  First, look at this comparison of two colosseum images:

![colosseum few matches](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_colosseum_best_match.jpg)

There were only about twelve matches, but they are dead on.  Then, when I looked at Big Ben, I saw a similar response with just eight matches:

![big ben few matches](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_bigben_best_match_no_nms.jpg)

It matched the faces of the clock!  At this point, I decided I would actually focus on just a few points, ignore NMS and use just those for determing what my matches would be.  I did bump it up to 200, because I decided it would make sense to use RANSAC here (after getting it working in part 3), and that would cause me to lose some points.

## Implementing RANSAC - Stealing from the Future
After I did Part 3, I realized that I could use my RANSAC algorithm from Part 3, to clean-up my matches in Part 1.  So, skip ahead to Part 3 if you want to know how I implemented RANSAC.  Here, you can see see the results of my eiffel tower matches, starting with 200 feaures, no NMS, and then without RANSAC implemented:

![eiffel without RANSAC](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_eiffel.jpg)

And then with RANSAC implemented.

![eiffel with RANSAC](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/matching_test_ransac_eiffel.jpg)

All the bad matches are gone.  RANSAC filtered them out. I can only visually see one outlier there left.  It even successfully matches the streetlights to the bottom left of the Eiffel tower.  At this point, I was finally happy with my matches.

## How should I value connections?
Initially, I went with a very simple (dumb?) heurestic to see how two images were related, and that was the number of connections.  In fact, you'll see my dictionary of edge weights referred to as "common\_points\_matrix".  It was a terribly metric, but was good for doing quick testing to see if my algorithm would work.  The results were not any better than randomly placing them into different centroids.  In fact, it wasn't even that good at times, because it would tend to almost all of the images in one clusters, and then leave the other clusters with just two or three images, and sometimes just the centroid.

After I implemented my decisions to use fewer features, not use NMS, and then filter with RANSAC, I'm confident that most of my matches will be high quality matches.  At the point, I changed my metric to average_distance.   Since in some cases I only had a few connections, but they were all high quality matches. I would want them to stand out.  So, I summed all my distances for my matches and then divided by the number of matches.  For images with no matches, I default set it to 1000 to ensure that those images are never actually considered a strong match.

## How should I cluster?
In order to do the clustering, I used k-means clustering, as suggested in the assignment prompt.  To be honest, I was entirely crazy about it, because even though my centroids could change, I never saw them adjust more than a couple of runs, and what those initial centroids were impacted the results significantly.  The biggest issue I had was that results tended to clump together.  If I was grading this, I would recommend running each program several times in order to average out the clustering results for a ballpark accuracy. 

## Results
Most surprising to me, is that I seemed to get results around 80%.  Which seemed really high to me.  However, the way we are measuring, tends to favor true negatives.  If images end up more or less evenly distributed across the different clusters, you will get about 80% of them correct.  That's just a function of the fact that an image can only have 8 to 9 true positives, while it can have about 80 true negatives.  If it is in a  group of only about 10 to 15, that's only 14 fewer True Positives. So, if we don't get a huge clump in one cluster, then the accuracy will be in the ballpark of 80 percent.  For example, if we randonly placed the 93 images into a 10 groups, that would end up with a decent accuracy.

## Recommendations for Improving Part 1

### Different Clustering Techniques
I used k-means for my clustering.  The biggest disadvantage of k-means is that a lot depends on the initial centroids.  You data could get really out of whack if it was given particularly odd centroids, or all from one class. If given more time, I think implementing something like [Single-linkage clustering](https://en.wikipedia.org/wiki/Single-linkage_clustering) would be better, as it gets around the issue of the initial centroids. There are a dozen different ways probably to do it, and given enough time, I would test them all, and would look at ensemble methods of them,

### Different heurestic for connections
Similarily, I don't know if average distance is the best measurement.  Perhaps, it's the probability that something is a corner?  There are probbly countless ways to measure their connection. I would definitely look at different factors in the future for measuring them.

### Better Accuracy Method
The method implemented for accuracy has a strong bias to True Negatives, and favors just about any system that results in well distributed groups. 

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

