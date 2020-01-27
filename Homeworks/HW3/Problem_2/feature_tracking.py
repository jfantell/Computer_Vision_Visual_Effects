from enum import Enum
import cv2
import numpy as np
from scipy.stats.stats import pearsonr


def get_bounds(x, y, descriptor_offset, search_offset):
    # For descriptor blocks, search_offset should be 1
    # This defines bounds of sliding window
    top = y - int(descriptor_offset * search_offset) - 1
    bottom = y + int(descriptor_offset * search_offset)
    left = x - int(descriptor_offset * search_offset) - 1
    right = x + int(descriptor_offset * search_offset)
    return top, bottom, left, right


def adjust_bounds(top, bottom, left, right, image_height, image_width):
    if top < 0:
        top = 0
    if bottom > image_height:
        bottom = image_height
    if left < 0:
        left = 0
    if right > image_width:
        right = image_width
    return top, bottom, left, right


def custom_descriptor(intensities):
    # Compute image derivatives
    dX = cv2.Sobel(intensities, cv2.CV_64F, 1, 0, ksize=1)
    dY = cv2.Sobel(intensities, cv2.CV_64F, 0, 1, ksize=1)

    # Compute gradient magnitude
    GM = np.sqrt(dX ** 2 + dY ** 2)

    # Compute gradient direction
    GD = np.arctan2(dY, dX)

    # Binned Gradient Orientation
    return np.concatenate([intensities, GM, GD], axis=1)


def compute_similarity(x_prev, y_prev, curr_frame, curr_keypoints, descriptor_offset, prev_frame_descriptor,
                       similarity_mode):
    min_sum_squares = float('inf')  # want to minimize this
    max_ncc = 0  # want to maximize this
    x_curr = x_prev
    y_curr = y_prev

    curr_keypoints = curr_keypoints.tolist()
    for (i, j) in curr_keypoints:
        top, bottom, left, right = get_bounds(i, j, descriptor_offset, 1)
        # top, bottom, left, right = adjust_bounds(top, bottom, left, right, prev_frame.shape[0], prev_frame.shape[1])
        curr_frame_intensities = curr_frame[top:bottom, left:right]
        curr_frame_descriptor = custom_descriptor(curr_frame_intensities)

        # flatten both descriptors (create 1d vector)
        curr_frame_descriptor = curr_frame_descriptor.flatten()
        prev_frame_descriptor = prev_frame_descriptor.flatten()

        if similarity_mode == "ssd":
            # Compute sum of squared diff
            sum_squares_tmp = np.sum((curr_frame_descriptor - prev_frame_descriptor) ** 2)
            if sum_squares_tmp < min_sum_squares:
                min_sum_squares = sum_squares_tmp
                x_curr = i
                y_curr = j
        elif similarity_mode == "ncc":
            # Compute Normalized Cross Correlation
            curr_frame_descriptor = (curr_frame_descriptor - np.mean(curr_frame_descriptor)) / (
                np.std(curr_frame_descriptor))
            prev_frame_descriptor = (prev_frame_descriptor - np.mean(prev_frame_descriptor)) / (
                np.std(prev_frame_descriptor))
            ncc_tmp = pearsonr(curr_frame_descriptor, prev_frame_descriptor)
            if ncc_tmp[0] > max_ncc:
                max_ncc = ncc_tmp[0]
                x_curr = i
                y_curr = j
        else:
            print("Please enter valid similarity poisson_mode")
    return x_curr, y_curr


def find_points(prev_frame, curr_frame, prev_frame_points, detector, similarity_mode):
    global curr_points, display_keypoints

    curr_points = []
    display_keypoints = []
    for idx, (x_prev, y_prev) in enumerate(prev_frame_points):
        # Create block of image intensities
        # using neighboring pixels around each
        # previously identified corner point

        #20 works for bed scene
        descriptor_offset = 20
        search_offset = .5

        # Get bounds of block
        top, bottom, left, right = get_bounds(x_prev, y_prev, descriptor_offset, 1)
        # Adjust the bounds
        # top, bottom, left, right = adjust_bounds(top,bottom,left,right,prev_frame.shape[0], prev_frame.shape[1])

        # Get descriptor for previous image
        prev_frame_intensities = prev_frame[top:bottom, left:right]
        prev_frame_descriptor = custom_descriptor(prev_frame_intensities)
        print("SHAPE",prev_frame_descriptor.shape)

        # Define bounds of search area
        top, bottom, left, right = get_bounds(x_prev, y_prev, descriptor_offset, search_offset)

        # Adjust the bounds
        # top, bottom, left, right = adjust_bounds(top,bottom,left,right, prev_frame.shape[0], prev_frame.shape[1])

        # Get search window
        search_window = curr_frame[top:bottom, left:right]

        # Compute keypoints
        keypoints = None
        if detector == 'harris':
            harris_corners = compute_harris(search_window)

            # Threshold harris corners
            keypoints = np.argwhere(harris_corners > .7 * harris_corners.max())

            # Recall numpy arrays use y,x indexing
            keypoints = np.flip(keypoints, axis=1)
        elif detector == 'orb':
            keypoints = compute_orb(search_window)

        if len(keypoints) == 0:
            print("No keypoints could be found near ({},{})".format(x_prev, y_prev))
            continue

        keypoints_adjusted = np.zeros_like(keypoints)
        keypoints_adjusted[:, 0] = x_prev - int(search_offset * descriptor_offset) + keypoints[:, 0]
        keypoints_adjusted[:, 1] = y_prev - int(search_offset * descriptor_offset) + keypoints[:, 1]

        # Visualize all keypoints
        display_keypoints.extend(keypoints_adjusted.tolist())

        # Slide window throughout search area of size equal
        # to feature descriptor block
        x_curr, y_curr = compute_similarity(x_prev, y_prev, curr_frame, keypoints_adjusted, descriptor_offset,
                                            prev_frame_descriptor, similarity_mode)
        curr_points.append([x_curr, y_curr])
    return curr_points


def compute_harris(window):
    gray_frame = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    # result is dilated for marking the corners, not important
    harris_frame = cv2.cornerHarris(gray_frame, 5, 3, 0.04)
    harris_frame = cv2.dilate(harris_frame, None)
    return harris_frame


def compute_orb(window):
    detector = cv2.ORB_create(edgeThreshold=0)
    keypoints_ = detector.detect(window, None)  # list of keypoint objects, get raw indicies
    keypoints = []
    for kp in keypoints_:
        x, y = kp.pt
        keypoints.append([int(x), int(y)])
    print("Number of ORB keypoins found: {}".format(len(keypoints)))
    return np.array(keypoints)


def draw_point(frame, x, y, color, radius):
    cv2.circle(frame, (x, y), radius, color, -1)


def draw_points(frame, points, color, radius):
    for (x, y) in points:
        draw_point(frame, x, y, color, radius)


current_frame_gui = None
clicked_points = []
display_keypoints = []


class Modes:
    MOVIE = 1
    IMAGE = 2
    OTHER = 3

# POINTS SHOULD BE ADDED IN THE FOLLOWING ORDER:
#
#  TOP LEFT, TOP RIGHT, BOTTOM LEFT, BOTTOM RIGHT
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        draw_point(current_frame_gui, x, y, (0, 255, 0), 5)


def apply_point_offset(points):
    offset = 20
    points_offset = []

    # top left
    x, y = points[0]
    x = x - offset
    y = y - offset
    points_offset.append([x, y])

    # top right
    x, y = points[1]
    x = x + offset
    y = y - offset
    points_offset.append([x, y])

    # bottom left
    x, y = points[2]
    x = x - offset
    y = y + offset
    points_offset.append([x, y])

    # bottom rightharris-laplace
    x, y = points[3]
    x = x + offset
    y = y + offset
    points_offset.append([x, y])

    return points_offset


def create_text_bubble(points, frame, bubble_text_queue, bubble_text_bin):
    # Height and width
    H = frame.shape[0]
    W = frame.shape[1]

    # Find centroid of points
    c_x = 0
    c_y = 0
    for p in points:
        c_x += p[0]
        c_y += p[1]
    c_x = c_x//len(points)
    c_y = c_y//len(points)

    cv2.circle(frame, (c_x,c_y), 20, (255,0,0), thickness=-1, lineType=8, shift=0)

    # Ellipse size
    ellipse_vertical_offset = -140
    ellipse_horizontal_offset = -70
    ellipse_major_axis_size = 200
    ellipse_minor_axis_size = 100

    # Centroid offset
    c_x += ellipse_horizontal_offset
    c_y += ellipse_vertical_offset

    # Adjust bounds (if needed)
    if c_x - ellipse_major_axis_size < 0:
        c_x = ellipse_major_axis_size
    elif c_x + ellipse_major_axis_size > W:
        c_x = W - ellipse_major_axis_size
    if c_y - ellipse_minor_axis_size < 0:
        c_y = ellipse_minor_axis_size
    elif c_y + ellipse_minor_axis_size > H:
        c_y = H - ellipse_minor_axis_size

    # ###### MANUALLY OVERRIDE CENTROID LOCATION
    # # i.e. no tracking, text stays in fixed location
    # c_x = 400
    # c_y = 700

    # Create overlay
    overlay = frame.copy()

    # https://docs.opencv.org/4.1.2/d6/d6e/group__imgproc__draw.html
    cv2.circle(overlay, (c_x, c_y), 20, (0, 0, 255), -1)


    # Change speaker bubble color based on who is speaking/texting
    speaker = bubble_text_queue[bubble_text_bin][0]
    message = bubble_text_queue[bubble_text_bin][1]
    bubble_color = (255, 255, 51)
    if(speaker == "John"):
        bubble_color = (100,0,255)
    cv2.ellipse(overlay, (c_x, c_y), (ellipse_major_axis_size, ellipse_minor_axis_size), 0, 0, 360, bubble_color, -1)
    cv2.ellipse(overlay, (c_x, c_y), (ellipse_major_axis_size, ellipse_minor_axis_size), 0, 0, 360, (0, 0, 255), 4)

    # https://stackoverflow.com/questions/27647424/opencv-puttext-new-line-character
    text = "{}:\n{}".format(speaker,message)
    text_vertical_offset = int(-ellipse_minor_axis_size * .55)
    text_horizontal_offset = int(-ellipse_major_axis_size * .6)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .7
    thickness = 1

    textHeight = cv2.getTextSize(text,fontFace,fontScale,thickness)[0][1]

    # For simulating newlines
    dy = textHeight + 10

    # Insert text
    c_x += text_horizontal_offset
    c_y += text_vertical_offset
    for i, line in enumerate(text.split('\n')):
        cv2.putText(overlay, line, (c_x, c_y + i * dy), fontFace, fontScale, thickness)

    # alpha blend overlay with frame
    alpha = 0.8
    frame = alpha * overlay + (1-alpha) * frame
    return frame

def create_warp_comosite(composite_image,curr_frame_points_offset,current_frame):
    # Create point correspondences for perspective transformation
    curr_frame_points_offset_array = np.array(curr_frame_points_offset).astype(np.float32)
    input_image_boundary_points_array = np.array(
        [(0, 0), (composite_image.shape[1], 0), (0, composite_image.shape[0]),
         (composite_image.shape[1], composite_image.shape[0])], dtype=np.float32)

    M = cv2.getPerspectiveTransform(input_image_boundary_points_array, curr_frame_points_offset_array)
    maxWidth = current_frame.shape[1]
    maxHeight = current_frame.shape[0]

    # Warp composite image using perspective transformation matrix
    warped = cv2.warpPerspective(composite_image, M, (maxWidth, maxHeight))

    # use warped as mask to superimpose warped on current background
    # frame
    mask = (warped == [0, 0, 0]).all(-1)
    assert (current_frame.shape == composite_image.shape)
    current_frame_output_composite = np.where(mask[..., None], current_frame, warped)
    return current_frame_output_composite

def main():
    # Open and save video files using unique path id
    scene = "bed_scene"
    warp_flag = False
    bubble_flag = True
    # Used for text scene
    #bubble_text_queue = [("Hayley","Evil interdimensional\nmonsters are attacking\ncampus"),("Hayley","Snevy needs us to\ndefeat their boss,\nThe GOLIATH"),("Hayley","So the monsters can\ngo back to their\nown dimension"),("John","I'm in! (For Snevy)\n"),("Hayley","Great! Okay, run\nto the VCC! Be careful\n...monsters around")]
    bubble_text_queue = [("Snevy","A giant scary\nmonster is attacking!\nCan you help me\ndefeat it?"),("Snevy","Thank you!")]

    ###### TRACKING VIDEO
    # Open video stream to input movie
    tracking_video = "inputs/tracking_videos/{}.MOV".format(scene)
    track_cap = cv2.VideoCapture(tracking_video)
    start_frame = 0
    # Get metadata from input movie
    track_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = track_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(track_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(track_cap.get(3))
    frame_height = int(track_cap.get(4))

    ###### OUTPUT VIDEO
    # Define the codec and create VideoWriter object
    # to write a new movie to disk
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_composite = cv2.VideoWriter('outputs/video_output_composite_{}.MOV'.format(scene), fourcc, fps,
                                    (frame_width, frame_height))
    out_tracking = cv2.VideoWriter('outputs/video_output_tracking_{}.MOV'.format(scene), fourcc, fps,
                                   (frame_width, frame_height))

    ###### Composite Input
    mode = Modes.MOVIE

    # Choose to composite a video or image into the tracked planar object
    composite_cap = None
    composite_image = None

    if mode == Modes.IMAGE:
        composite_image = cv2.imread("inputs/composite_images/brick_wall.JPG")
    elif mode == Modes.MOVIE:
        composite_cap = cv2.VideoCapture("inputs/composite_videos/space.mp4")

    ####### MANUALLY SELECT POINTS
    # Set mouse clip
    cv2.namedWindow("Create Features To Track")
    cv2.setMouseCallback("Create Features To Track", click)

    # Get a video frame and select 4 points
    # From here on out all frames will be tracked
    ret, current_frame = track_cap.read()
    if not ret:
        print("Unable to process first frame...terminating script")
        return 1

    # Save copy of frame into global variable (for use in callback function)
    global current_frame_gui
    current_frame_gui = current_frame.copy()

    # Allow user to select bounding box points
    while True:
        # Display image
        cv2.imshow('Create Features To Track', current_frame_gui)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit")
            cv2.destroyAllWindows()
            break
        if key == ord('u'):  # erase points
            global clicked_points
            clicked_points = []
            current_frame_gui = current_frame.copy()

    if len(clicked_points) != 4:
        print("In order to apply the perspective transform you must select exactly 4 points")

    # Write first frame to disk
    out_composite.write(current_frame_gui)
    out_tracking.write(current_frame_gui)

    ###### FEATURE TRACKING FRAMES
    print("User added the following points {}".format(clicked_points))
    print("Starting Harris Corner Tracking...")
    prev_frame = current_frame
    prev_frame_points = clicked_points

    frame_index = 0

    ### Required for text bubble
    bubble_text_bin = 0
    swap_text_index = frame_count//max(len(bubble_text_queue),1) #avoid division by 0
    while track_cap.isOpened():
        # Get frame from Tracking video
        ret, current_frame = track_cap.read()
        if not ret:
            print("Unable to process frame...terminating script")
            return 1

        # Get frame from composite video
        if mode == Modes.MOVIE:
            ret, composite_image = composite_cap.read()

        # Compute sum of squared diff between current and prev harris images
        curr_frame_points = find_points(prev_frame, current_frame, prev_frame_points, "orb", "ncc")

        # Display points and keypoints
        current_frame_output = current_frame.copy()
        draw_points(current_frame_output, display_keypoints, (0, 0, 255), 5)
        draw_points(current_frame_output, curr_frame_points, (0, 255, 0), 3)

        # Apply perspective transform and composite image/video on top of tracked points
        current_frame_output_composite = current_frame.copy()
        if len(curr_frame_points) == 4 and warp_flag:
            # add small constant to each pixel in input image to ensure that
            # the image has no pure black pixels
            # this is neccessary to utilize the input image
            # as a mask when inserting it onto a movie
            # frame
            # ENSURE THE IMAGE PIXEL INTENSITIES DO NOT OVERFLOW (UINT8)
            composite_image[composite_image != 255] += 1

            # Resize image such that it is the same as the video resolution
            composite_image = cv2.resize(composite_image, (frame_width, frame_height))

            # Apply offset to cover frame markers
            curr_frame_points_offset = apply_point_offset(curr_frame_points)
            current_frame_output_composite = create_warp_comosite(composite_image,curr_frame_points_offset,current_frame_output_composite)

        # Create text bubble
        if bubble_flag and len(bubble_text_queue) >= 1 and len(curr_frame_points) >= 1:
            # if (frame_index % swap_text_index == 0) and (bubble_text_bin != len(bubble_text_queue) -1):
            if frame_index == 240:
                bubble_text_bin += 1
            current_frame_output_composite = create_text_bubble(curr_frame_points,
                                                                current_frame_output_composite, bubble_text_queue, bubble_text_bin)

        # Convert frame to uint8
        current_frame_output_composite = current_frame_output_composite.astype(np.uint8)

        # Display the frame for diagnostic purposes
        cv2.imshow('Final Frame', current_frame_output_composite)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Write frame to disk
        out_composite.write(current_frame_output_composite)
        out_tracking.write(current_frame_output)

        # Set current frame to previous frame
        prev_frame = current_frame
        prev_frame_points = curr_frame_points
        frame_index += 1
        print(frame_index)

    cv2.destroyAllWindows()
    print("Finished Processing All Frames")
    return 0


if __name__ == "__main__":
    main()
