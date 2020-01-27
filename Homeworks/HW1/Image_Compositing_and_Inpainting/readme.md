Task:

Write an implementation of Poisson image compositing that takes as inputs an RGB
source image S, an RGB target image T , and a binary mask M that is 1 at pixels of S that should
appear in the composite. In addition to the description in Section 3.2.1, the small example in
Problem 3.8 may help in figuring out the necessary linear system that should be solved. You can
use any language you like; if you use Matlab, the command delsq may be useful.
1. Demonstrate your implementation on at least 3 (source, target) pairs from Activity 1 (or additional images you take on your own). Critically assess the results.
2. Further implement the mixed-gradient approach to avoid the problem illustrated in Figure
3.10b. Demonstrate before-and-after images where using mixed gradients improves the composite
