# CS B657 - Assignment 2: Warping, Matching, Stitching, Blending

Completed by Derrick Eckardt on March 19, 2019.  Please direct any questions to [derrick@iu.edu](mailto:derrick@iu.edu)

The assignment prompt can be found at [Assignment 2 Prompt](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/a2-sp2019.pdf)

# General Comments
I think the coolest thing about this assignment, is that it becomes really easy to understand how our a lot of our every day technology actually works.  By the end, I've taken two different images of the same thing, summed them together, and got some startinly cool results!  Most interesting, was the cyclical nature of this assignment.  After I did part 1, and part 2, I was able to use things I learned or coded in part 3 to go back and improve part 1 and part 2. I have a lot else to say on specific points, so let's get on with the show.

# Part 1 - Image matching and clustering
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
Most surprising to me, is that I seemed to get results around 77-85%.  Which seemed really high to me.  However, the way we are measuring, tends to favor true negatives.  If images end up more or less evenly distributed across the different clusters, you will get about 80% of them correct.  That's just a function of the fact that an image can only have 8 to 9 true positives, while it can have about 80 true negatives.  If it is in a  group of only about 10 to 15, that's only 14 fewer True Positives. So, if we don't get a huge clump in one cluster, then the accuracy will be in the ballpark of 80 percent. For example, if we randonly placed the 93 images into a 10 groups, that would end up with a decent accuracy, and each image was in a group with none of colleagues, we still get ~75 TN for each.

## Recommendations for Improving Part 1

### Different Clustering Techniques
I used k-means for my clustering.  The biggest disadvantage of k-means is that a lot depends on the initial centroids.  You data could get really out of whack if it was given particularly odd centroids, or all from one class. If given more time, I think implementing something like [Single-linkage clustering](https://en.wikipedia.org/wiki/Single-linkage_clustering) would be better, as it gets around the issue of the initial centroids. There are a dozen different ways probably to do it, and given enough time, I would test them all, and would look at ensemble methods of them,

### Different heurestic for connections
Similarily, I don't know if average distance is the best measurement.  Perhaps, it's the probability that something is a corner?  There are probbly countless ways to measure their connection. I would definitely look at different factors in the future for measuring them.

### Better Accuracy Method
The method implemented for accuracy has a strong bias to True Negatives, and favors just about any system that results in well distributed groups.  Perhaps, all we really only care about the True Positives.  That would yield drastically different results, although that would now favor all points being in one big cluster.  So, there's are other solutions, but they all have their issues, and probably work best for specific cases. Definitely something worth exploring in the future.

### Code Refactor
With the amount of steps and things going on, there must be ways to optimize my code further.  I tried to used fast data structures like dictionaries as much as possible, which helped keep it low.  However, since this is the part of the assignment that takes even now about 25 minutes to run for the full test set, there are definitely opportunities to improve this.  Running RANSAC through my pairs is the most time intensive portion of this, but it is a step worth doing.  I would definitely find ways to improve my RANSAC for the future.

# Part 2 - Image transformations
This started with figuring out the mechanics of transforming this picture from the lincoln memorial:

![input lincoln](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/part2-images/lincoln.jpg)

into this image:

![warped lincoln](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/part2-images/lincoln_test.jpg)

That was a good introduction into how to do it.  Then, I worked on my book warping, should be.  Instead of starting at case 1, I actually ended up working backwards, and that worked out alright.

Honestly, I didn't have too many tough decisions to make here since how to do it was outlined fairly well in the assignment prompt.  I do have a few things to say, but the most relevant one is what size should my output image be, as it would creep up time and again.

## What size should my output image be?
This was something that early on, I thought would be straightforward, which like most items here, turned out to have some intricarcies.  Depending on the transform matrix applied, particularly for Rigid or Affine transformations, I would see my image get cut-off on the screen.

I attempted to resolve this with a function called get_shape(), which looked at the corners of the base image, forward warped them to determine what the bounding box would be, along with any x,y offsets to account for the fact that the warping could mean that it wanted a point from a negative spot.  This would also minimize the amount of white space.

However, when I implemented get_shape, what was lost was the fact that in comparison to the original image, it was no longer always obvious that something had happened in every dimension.  For example, the Rigid transformations now just looked like rotations, with the 

Ultiamtely, as a design decision, I decided to go with the original size of the image for cases, 2,3, and 4, so you could best see the difference in the transformation.  For case 1, I changed the shape so that the translated portion would appear black.  I left the function and offsets in place to improve on this in the future.

## Results
This was pretty cool, when I was able to get the two book images to rotate as it was meant to be rotated.

![rotated book - Tada!](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/part2_4_output.jpg)

If you flip between the book1 image and the new image, the book in the middle does not move.  What is interesting is that the stuff around it does, because those are items that you can't shift, such as a shadow.

## Recommendations for Improving Part 2
Here are some thoughts on how I could improve this part of my program.

### Better Visualize the Movement
As I stated in the "what size should my output image be?" section, I think there is a better way to visualize the transformations, than I have implemented.  If I had another cut at it, or use for this program, I would work further on this seemingly easy problem, that is not so easy.  

### Error Checking
I implemented almost no steps to make sure that the input points would not result in a non-invertible singular matrix.  Ideally, I could implement something that checked for a singular matrix, and then nudged the reference points a pixel or two in order to hopefully get a non-singular matrix.

### Refactoring
I cringe when I look at how the code is structured in part2().  However, with that part of the program only taking 10 seconds to run, there is little incentive to refactor the code for time.  It could be made clearer for readability.  However, as it is now, I think it makes it easier to compare the different types on matrices needed to perform the different kinds of transformations.

# Part 3 - Automatic image matching and transformations
This was by far the most interesting part where I combined elements from Part 1's image matching with Part2's transformation, and then being able to merge them into one photo

## RANSAC
RANSAC was the means I used in order to find the transformation matrix.  I started with the same process I had in part one to find matches.  Then, those matches were fed into the RANSAC function, which would give me the transformation matrix, and a revised list of matches that are inliers for that transformation.

First, I set some initial conditions.  The default for a transformation matrix would be the identity function.  This would only ever be used if every single iteration was found to be singular.

More importantly, I set my pixel threshold to be just three pixels.  This comes into play with counting inliers.   I would determine my inliers by picking four matches at random, and compute the transform matrix.  Then, I applied the transform matrix to the x,y in every match.  Which produced an x'o,y'o.  If those x'o,y'o were within three pixels in of the x',y' that were in the calculated matches, then that match was considered an inlier.  Otherwise, it was an outlier.

I kept track of how many inliers each matrix had, and kept the highest number of inliers as I repeated this process. I choose to do 500 iterations. Which is most cases, might be excessive, but not all of them.  Since, I was picking 4 points at a random.  For just 20 points, that would mean 116280 different combinations.  Way more than I could test in a reasonable amount of time.  Since RANSAC only works if the majority are inliers, than it meant that I had a pretty good chance of landing on an inlier heavy matrix.  I found 500 to work almost all the time, if an image would be matched up.  It took only about a quarter of second for the harder cases.

After 500 iterations, I passed the matrix and a revised matches list back to the program.  Earlier, I highlighted the before and after for the eiffel tower.

## Results
I think it's interesting that there are definitely still other items that would have to be corrected for, such as the fact that in one image, the table seems to be much larger than the other ones, so you end up with that table-in-table effect.

## Recommendations for Improving Part 3

### Image size
As I mentioned earlier, trying to get the right size of the image left a few things in limbo.  And depending on which two images were being merged, there could have been some edges that get cut off.


# Errata
I learned so much about python and numpy data structures.  My favorite error is when I was trying to merge the images and average out the results.   As it turns out, if you attempt to add two 8-bit numpy integers, you get non-sensical answers, which result in merged images like the following:

![oops](https://github.iu.edu/cs-b657-sp2019/derrick-a2/blob/master/part3_book_test_bad_colors.jpg)


