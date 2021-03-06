% Auto-generated by cameraCalibrator app on 31-Oct-2019
%-------------------------------------------------------


% Define images to process
imageFileNames = {'/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1353.JPG',...
    '/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1354.JPG',...
    '/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1355.JPG',...
    '/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1356.JPG',...
    '/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1357.JPG',...
    '/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1358.JPG',...
    '/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1361.JPG',...
    '/Volumes/2nd Storage/Development/Computer Vision Visual Effects/Homeworks/HW3/problem1/DSLR/IMG_1363.JPG',...
    };
% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
squareSize = 33;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams,'BarGraph');
h2=figure; showReprojectionErrors(cameraParams,'ScatterPlot');

% Visualize pattern locations
h3=figure; showExtrinsics(cameraParams, 'CameraCentric');
h4=figure; showExtrinsics(cameraParams, 'PatternCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')
