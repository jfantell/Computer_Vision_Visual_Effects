%% CHOOSE EXAMPLE
% all examples are stored in the following format
% Example/<#>/<file>
example_num = 1;

%% LOAD AND RESIZE IMAGES %%
I1 = imread(sprintf('Examples/Morphing/%d/I1.JPG', example_num));
I2 = imread(sprintf('Examples/Morphing/%d/I2.JPG', example_num));
I1 = imresize(I1, [NaN 1000]);
I2 = imresize(I2, [NaN 1000]);

% SANITY CHECK: make sure images are of same size
assert((size(I1,1) == size(I2,1)) & (size(I1,2) == size(I2,2)));

% Record height and width of images
M = size(I1,1);
N = size(I1,2);

%% CHOOSE GROUND TRUTH CORRESPONDENCES
%%% WARNING: THIS BLOCK SHOULD ONLY BE UNCOMMENTED IF THE NEXT BLOCK IS
%%% COMMENTED OUT (AND VICE VERSA)
% [I1_points,I2_points] = cpselect(I1,I2,'Wait',true);
% I1_points = [I1_points; [1,1]; [1,N];[M,1];[M,N]];
% I2_points = [I2_points; [1,1]; [1,N];[M,1];[M,N]];
% save(sprintf('Examples/Morphing/%d/I1_points_.mat',example_num),'I1_points');
% save(sprintf('Examples/Morphing/%d/I2_points_.mat',example_num),'I2_points');

%% LOAD (PRE-CHOSEN) GROUND TRUTH CORRESPONDENCES %%
%%% WARNING: THIS BLOCK SHOULD ONLY BE UNCOMMENTED IF THE NEXT BLOCK IS
%%% COMMENTED OUT (AND VICE VERSA)
I1_points_stuct = load(sprintf('Examples/Morphing/%d/I1_points_.mat',example_num));
I2_points = I1_points_stuct.I1_points;
I2_points_struct = load(sprintf('Examples/Morphing/%d/I2_points_.mat',example_num));
I1_points = I2_points_struct.I2_points;

%% COMPUTE FORWARD AND BACKWARD FLOW VECTORS
weights_I1_to_I2 = get_weights(I1_points,I2_points);
coordinates_I1 = create_fine_grid(I1,1);
coordinates_I1_warped = compute_coorespondences(coordinates_I1,I1_points,weights_I1_to_I2);
% coordinates_I1_warped = clip_points(coordinates_I1_warped,M,N);
fwd_vector = coordinates_I1_warped - coordinates_I1;
%bwd_vector = -fwd_vector; % let the backward vector be the opposite of the forward vector

% create sparse grid from Image 1 (every 3rd pixel or so)
weights_I2_to_I1 = get_weights(I2_points,I1_points);
coordinates_I2 = create_fine_grid(I2,1);
coordinates_I2_warped = compute_coorespondences(coordinates_I2,I2_points,weights_I2_to_I1);
bwd_vector = coordinates_I2_warped - coordinates_I2;

% sparse_coordinates_I2 = create_fine_grid(I2,1);
% sparse_coordinates_I2_warped = compute_coorespondences(sparse_coordinates_I2,I2_points,weights_I2_to_I1);
% sparse_coordinates_I2_warped = clip_points(sparse_coordinates_I2_warped,M,N);
% bwd_vector = sparse_coordinates_I2_warped - sparse_coordinates_I2;
% % bwd_vector = reshape(bwd_vector,[M,N,2]);


%% CREATE INTERMEDIATE IMAGES AND CROSS-DISOLVE
video = VideoWriter(sprintf('Examples/Morphing/%d/morph_video.mp4',example_num),'MPEG-4');
open(video);
I1_t = zeros(M,N,3);
I2_t = zeros(M,N,3);
M_t = zeros(M,N,3);
for t = 0:0.1:1
    vector_index = 1;
    for x = 1:N
        for y = 1:M
            v = round(t*fwd_vector(vector_index,2));
            u = round(t*fwd_vector(vector_index,1));

            new_point = [x + u,y + v];
            new_point = clip_points(new_point,M,N);

            I1_t(y,x,:) = I1(new_point(2),new_point(1), :);

            v = round((1-t)*bwd_vector(vector_index,2));
            u = round((1-t)*bwd_vector(vector_index,1));

            new_point = [x + u,y + v];
            new_point = clip_points(new_point,M,N);
            I2_t(y,x,:) = I2(new_point(2),new_point(1), :);

            vector_index = vector_index + 1;
        end
    end
    M_t = (1-t) * I1_t + t * I2_t;
    M_t = uint8(M_t);
    writeVideo(video, M_t);
end
close(video);