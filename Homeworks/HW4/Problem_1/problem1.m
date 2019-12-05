%% CHOOSE EXAMPLE
% all examples are stored in the following format
% Example/<#>/<file>
example_num = 2;

%% LOAD AND RESIZE IMAGES %%
I1 = imread(sprintf('Examples/%d/I1.JPG', example_num));
I2 = imread(sprintf('Examples/%d/I2.JPG', example_num));
I1 = imresize(I1, [NaN 1000]);
I2 = imresize(I2, [NaN 1000]);

% SANITY CHECK: make sure images are of same size
assert((size(I1,1) == size(I2,1)) & (size(I1,2) == size(I2,2)));

% Record height and width of images
M = size(I1,1);
N = size(I1,2);

%% (PROBLEM 1A) CHOOSE GROUND TRUTH CORRESPONDENCES USING CPSELECT
%%% WARNING: THIS BLOCK SHOULD ONLY BE UNCOMMENTED IF EITHER OF THE NEXT
%%% TWO BLOCKS ARE COMMENTED OUT
% [I1_points_,I2_points] = cpselect(I1,I2,'Wait',true);
% I1_points = [I1_points; [1,1]; [1,N];[M,1];[M,N]];
% I2_points = [I2_points; [1,1]; [1,N];[M,1];[M,N]];
% save(sprintf('Examples/%d/I1_points_.mat',example_num),'I1_points');
% save(sprintf('Examples/%d/I2_points_.mat',example_num),'I2_points');

[I1_points_more,I2_points_more] = cpselect(I1,I2,'Wait',true);
I1_points = [I1_points; I1_points_more];
I2_points = [I2_points; I2_points_more];
save(sprintf('Examples/%d/I1_points_.mat',example_num),'I1_points');
save(sprintf('Examples/%d/I2_points_.mat',example_num),'I2_points');



%% LOAD (PRE-CHOSEN) GROUND TRUTH CORRESPONDENCES %%
%%% WARNING: THIS BLOCK SHOULD ONLY BE UNCOMMENTED IF EITHER THE NEXT OR
%%% PREVIOUS BLOCK ARE COMMENTED OUT
% % I1_points_stuct = load(sprintf('Examples/%d/I1_points_.mat',example_num));
% % I1_points = I1_points_stuct.I1_points;
% % I2_points_struct = load(sprintf('Examples/%d/I2_points_.mat',example_num));
% % I2_points = I2_points_struct.I2_points;

%% (PROBLEM 2D)
%%% WARNING: THIS BLOCK SHOULD ONLY BE UNCOMMENTED IF EITHER THE OF THE PREVIOUS
%%% BLOCKS ARE COMMENTED OUT
% Source: https://www.mathworks.com/help/vision/examples/uncalibrated-stereo-image-rectification.html
% Convert to grayscale.
% I1gray = rgb2gray(I1);
% I2gray = rgb2gray(I2);
% 
% blobs1 = detectSURFFeatures(I1gray, 'MetricThreshold', 2000);
% blobs2 = detectSURFFeatures(I2gray, 'MetricThreshold', 2000);
% 
% [features1, validBlobs1] = extractFeatures(I1gray, blobs1);
% [features2, validBlobs2] = extractFeatures(I2gray, blobs2);
% 
% indexPairs = matchFeatures(features1, features2, 'Metric', 'SAD', ...
%   'MatchThreshold', 5);
% 
% matchedPoints1 = validBlobs1(indexPairs(:,1),:);
% matchedPoints2 = validBlobs2(indexPairs(:,2),:);
% 
% figure; ax = axes;
% showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
% title(ax, 'Candidate point matches');
% legend(ax, 'Matched points 1','Matched points 2');
% 
% I1_points = matchedPoints1.Location;
% I2_points = matchedPoints2.Location;

%% (PROBLEM 2B) Implement the thin-plate spline interpolation algorithm from Section 5.2
%%% get weights required for estimating correspondences %%
weights = get_weights(I1_points,I2_points);

% select testing points (i.e. not part of ground truth points set)
[I1_test_points,I2_test_points] = cpselect(I1,I2,'Wait',true); 

% estimate points from first image on second image
I1_test_points_warped = compute_coorespondences(I1_test_points,I1_points,weights);
figure, imshow(I2);
hold on;

% plot ground truth points
plot(I2_test_points(:,1),I2_test_points(:,2),'Marker','o','MarkerSize',6,'MarkerEdgeColor','r','MarkerFaceColor','r','LineStyle','none');
% plot estimated points
plot(I1_test_points_warped(:,1),I1_test_points_warped(:,2),'Marker','*','MarkerSize',4,'MarkerEdgeColor','b','MarkerFaceColor','b','LineStyle','none');
hold off;

%% (PROBLEM 2C) Warp first image onto second image using thin-plate spline algorithm %%

% create sparse grid from Image 1 (every 3rd pixel or so)
sparse_coordinates_I1 = create_fine_grid(I1,3);

% colors at every point in sparse_coordinates_I1
[r_I1,g_I1,b_I1] = get_colors(sparse_coordinates_I1,I1);

% warp points in Image 1 to Image 2
sparse_coordinates_I1_warped = compute_coorespondences(sparse_coordinates_I1,I1_points,weights);

% create dense grid from Image 2
dense_coordinates_I2 = create_fine_grid(I2,1);

% interpolate
interpolated_image = interpolate(sparse_coordinates_I1_warped, dense_coordinates_I2, [r_I1,g_I1,b_I1], M, N);

% show interpolated image
figure, imshow(uint8(interpolated_image));