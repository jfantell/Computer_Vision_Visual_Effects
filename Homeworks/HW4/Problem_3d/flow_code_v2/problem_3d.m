im1 = imread('data/im_rect.jpg');
im2 = imread('data/im2_rect.jpg');
im1 = imresize(im1, [NaN 1000]);
im2 = imresize(im2, [NaN 1000]);
uv = estimate_flow_interface(im1, im2, 'classic+nl-fast');

% Display estimated flow fields
figure; subplot(1,2,1);imshow(uint8(flowToColor(uv))); title('Disparity Map');
subplot(1,2,2); plotflow(uv);   title('Vector plot');