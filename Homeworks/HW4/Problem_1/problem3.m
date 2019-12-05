%% CHOOSE EXAMPLE
% all examples are stored in the following format
% Example/<#>/<file>
example_num = 1;

%% LOAD AND RESIZE IMAGES %%
I1 = imread(sprintf('Examples/Epipolar/%d/I1_1.JPG', example_num));
I2 = imread(sprintf('Examples/Epipolar/%d/I2_1.JPG', example_num));
I1 = imresize(I1, [NaN 1000]);
I2 = imresize(I2, [NaN 1000]);

% SANITY CHECK: make sure images are of same size
assert((size(I1,1) == size(I2,1)) & (size(I1,2) == size(I2,2)));

% Record height and width of images
Height = size(I1,1);
Width = size(I1,2);

%% CHOOSE GROUND TRUTH CORRESPONDENCES USING CPSELECT %%
%%% WARNING: THIS BLOCK SHOULD ONLY BE UNCOMMENTED IF THE NEXT BLOCK IS
%%% COMMENTED OUT (AND VICE VERSA)
% [I1_points,I2_points] = cpselect(I1,I2,'Wait',true);
% save(sprintf('Examples/Epipolar/%d/I1_points_.mat',example_num),'I1_points');
% save(sprintf('Examples/Epipolar/%d/I2_points_.mat',example_num),'I2_points');

%% LOAD (PRE-CHOSEN) GROUND TRUTH CORRESPONDENCES %%
%%% WARNING: THIS BLOCK SHOULD ONLY BE UNCOMMENTED IF THE NEXT BLOCK IS
%%% COMMENTED OUT (AND VICE VERSA)
I1_points_stuct = load(sprintf('Examples/Epipolar/%d/I1_points_.mat',example_num));
I1_points = I1_points_stuct.I1_points;
I2_points_struct = load(sprintf('Examples/Epipolar/%d/I2_points_.mat',example_num));
I2_points = I2_points_struct.I2_points;

%% (PROBLEM 2A) Find Fundamental Matrix

% Rescale and center points (Step 1)
[I1_points_s, T1] = scale_and_recenter_points(I1_points);
[I2_points_s, T2] = scale_and_recenter_points(I2_points);

% construct nx9 matrix A (Step 2)
A = compute_A_matrix(I1_points_s,I2_points_s);

% Compute SVD of A (Step 3) and extract f from V
[U,D,V] = svd(A);
f = V(:,9);

% Reshape f (Step 4)
F_hat = reshape(f,3,3)';

% Compute SVD of F_hat; Set D(3,3) to 0, replace F_hat with U*D*V (Step 5)
[U,D,V] = svd(F_hat);
D(3,3) = 0;
F_hat = U*D*V';

% Recover fundamental matrix estimate (Step 6)
F = T2' * F_hat * T1;
F_MATLAB = estimateFundamentalMatrix(I1_points,I2_points,'Method','Norm8Point');

% Ensure that p2 * Fp1 ~ 0 for all p2
mean_custom = test_f_matrix(F,I1_points,I2_points);
mean_matlab = test_f_matrix(F_MATLAB,I1_points,I2_points);
disp(["MatLab F mean",mean_matlab,"Custom F mean",mean_custom]);

plot_epipolar_lines(I1,I2,I1_points,I2_points,F_MATLAB);

%% (PROBLEM 2B) Apply an appropriate projective transformation/Rectify images %%
% compute eigenvalues and eigenvectors of F and F' to find epipoles
[V1,D1] = eig(F);
[V2,D2] = eig(F');

% extract eigenvalues
D2 = [D2(1,1) D2(2,2) D2(3,3)];
e2 = V2(:,2); %epipole is eigenvector with eigenvalue = 0
e2 = e2/e2(3); % convert from homogenous coordinates to cartesian coordinates

% Step 2: factorize M
% e1_x = [[0, -e1(3), e1(2)];[e1(3),0,e1(1)];[-e1(2),-e1(1),0]];
e2_x = [[0, -e2(3), e2(2)];[e2(3),0,e2(1)];[-e2(2),-e2(1),0]];
v = [1,1,1];
lambda = 1;
%P2 = [e2_x*F+e2*v, lambda*e2];
%M is left 3x3 submatrix of P2 per Professor's Piazza note
M = e2_x*F+e2*v;
%M = M/norm(M);

% SANITY CHECK: F = [e2]_x * M
disp("e2_x * M");
% disp((e2_x * (M/norm(M))));
disp(-e2_x * M);
% disp((e2_x * M)/norm(e2_x * M));
disp("F");
disp(F);
% disp("F_MATLAB");
% disp(F_MATLAB);

% Step 3: create Translation matrix T
x2_origin = round(Width/2);
y2_origin = round(Height/2);
T = [[1,0,-x2_origin];[0,1,-y2_origin];[0,0,1]];

% Step 4: create Rotation matrix R
Theta = atan(-e2(2)/e2(1));
R = [[cos(Theta),-sin(Theta),0];[sin(Theta),cos(Theta),0];[0 0 1]];
x_y_z_star = (R*T*e2);
x_star = x_y_z_star(1);

% Step 5: Homography
H2 = [[1,0,0];[0,1,0];[-1/x_star,0,1]] * R*T;

% figure(3);
% tformH2 = projective2d(H2');
% I2_warped = imwarp(I2,tformH2);
% imshow(I2_warped);
% title("Warped Image");

% Step 6: Apply projective transformation to I1 features
% I1_points_rect = (H2 * M * [I1_points,ones(size(I1_points,1),1)]')';
% I2_points_rect = (H2 * [I2_points,ones(size(I2_points,1),1)]')';
% I1_points_rect = I1_points_rect(:,1:2); % drop last column
% I2_points_rect = I2_points_rect(:,1:2);
% plot_epipolar_lines(I1,I2,I1_points_rect,I2_points_rect,F);

%% (PROBLEM 2C) Implement a basic stereo correspondence algorithm
% SOURCE: https://www.mathworks.com/help/vision/ref/disparitysgm.html#d117e148343
[t1, t2] = estimateUncalibratedRectification(F_MATLAB, I1_points, I2_points, size(I2));
tform1 = projective2d(t1);
tform2 = projective2d(t2);

% Rectified images (manually display via command line)
I1_warped = imwarp(I1,tform1);
% I1_points_rect = zeros(size(I1_points,1),2);
% for i = 1:size(I1_points,1)
%     tmp_point = t2 * [I1_points(i,:),1]';
%     I1_points_rect(i,:) = [tmp_point(1), tmp_point(2)];
% end

% (t1 * [I1_points,ones(size(I1_points,1),1)]')';

I2_warped = imwarp(I2,tform2);
% I2_points_rect = zeros(size(I2_points,1),2);
% for i = 1:size(I1_points,1)
%     tmp_point = t1 * [I2_points(i,:),1]';
%     I2_points_rect(i,:) = [tmp_point(1), tmp_point(2)];
% end

% I2_points_rect = (t2 * [I2_points,ones(size(I2_points,1),1)]')';
% plot_epipolar_lines(I1_warped,I2_warped,I2_points_rect(:,1:2),I1_points_rect(:,1:2),F);

[I1Rect, I2Rect] = rectifyStereoImages(I1, I2, tform1, tform2);

figure(4);
imshow(stereoAnaglyph(I1Rect, I2Rect));
title('Rectified Stereo Images');
% 
J1 = rgb2gray(I1Rect);
J2 = rgb2gray(I2Rect);
disparityRange = [64 192];
disparityMap = disparitySGM(J1,J2,'DisparityRange',disparityRange,'UniquenessThreshold',15);

figure(5);
imshow(disparityMap,disparityRange);
title('Disparity Map');
colormap jet;
colorbar;

% Source: https://inside.mines.edu/~whoff/courses/EENG512/lectures/26-Fundamental.pdf
function [points_rescaled, T_matrix] = scale_and_recenter_points(points_in)
    t = mean(points_in); % centroid of points
    points_centered = points_in - t; % shift the origin to 0,0
    average_distance = mean(sqrt(points_centered(1)^2 + points_centered(2)^2)); % compute the average euclidean distance across all points to the origin
    s = sqrt(2)/average_distance; % compute scale factor, so that average distance is sqrt(2)
    T_matrix = [s*eye(2),(-s*t)'; 0 0 1]; % transformation matrix (scaling plus translation)
    points_rescaled = points_centered * s;
end

function A = compute_A_matrix(point_set_1, point_set_2)
    x1 = point_set_1(:,1);
    x2 = point_set_2(:,1);
    y1 = point_set_1(:,2);
    y2 = point_set_2(:,2);
    A = [x2.*x1  x2.*y1 x2 y2.*x1 y2.*y1 y2 x1 y1 ones(size(point_set_1,1),1)];
end

function line_ = compute_epipolar_line(F_Matrix, point, width_i2)
    % Define domain of line (0 - max width image 2)
    x = linspace(0,width_i2,width_i2);
    
    % Convert cartesian points to homogenous points
    point = [point, 1];
    
    % Compute equation of line
    line_coefficients = F_Matrix * point';
    
    % Extract y; y = (-a*x - c)/b;
    y = (-line_coefficients(1)*x - line_coefficients(3))/line_coefficients(2);
    line_ = [x; y]';     
end

function mean_ = test_f_matrix(F,I1_points,I2_points)
    % Convert points to homogenous points
    num_points_ = size(I1_points,1);
    I1_points = [I1_points, ones(num_points_,1)];
    I2_points = [I2_points, ones(num_points_,1)];
    sum_ = 0;
    for i=1:num_points_
        tmp_ = I2_points(i,:) * F * I1_points(i,:)';
        sum_ = sum_ + tmp_;
    end
    mean_ = sum_/num_points_;
end

function plot_epipolar_lines(I1,I2,I1_points,I2_points,F)
    figure(1); 
    imshow(I1); hold on;
    title('First Image'); 
    plot(I1_points(:,1),I1_points(:,2),'Marker','*','MarkerSize',4,'MarkerEdgeColor','r','MarkerFaceColor','b','LineStyle','none');
    for i = 1:size(I1_points,1)
        line_ = compute_epipolar_line(F',I2_points(i,:),size(I1,2));
        line(line_(:,1),line_(:,2),'LineWidth',1);
    end

    % Project points from Image 1 onto lines in Image 2
    figure(2);
    imshow(I2); hold on;
    title('Second Image');
    plot(I2_points(:,1),I2_points(:,2),'Marker','*','MarkerSize',4,'MarkerEdgeColor','r','MarkerFaceColor','b','LineStyle','none');
    for i = 1:size(I1_points,1)
        line_ = compute_epipolar_line(F,I1_points(i,:),size(I2,2));
        line(line_(:,1),line_(:,2),'LineWidth',1);
    end
end