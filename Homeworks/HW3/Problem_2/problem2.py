from enum import Enum
import cv2
import numpy as np


def get_bounds(x, y, descriptor_offset, search_offset):
    # For descriptor blocks, search_offset should be 1
    # This defines bounds of sliding window
    top = y - int(descriptor_offset * search_offset)
    bottom = y + int(descriptor_offset * search_offset)
    left = x - int(descriptor_offset * search_offset)
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
    GM = np.sqrt(dX**2 + dY**2)

    # Compute gradient direction
    GD = np.arctan2(dY,dX)

    # Binned Gradient Orientation
    return np.concatenate([intensities,GM,GD],axis=1)

def sum_of_square_diff(prev_frame, curr_frame, prev_frame_points):
    global curr_points, harris_points

    curr_points = []
    harris_points = []
    for idx, (x, y) in enumerate(prev_frame_points):
        # Create block of image intensities
        # using neighboring pixels around each
        # previously identified corner point
        descriptor_offset = 20
        search_offset = 1.05

        # Get bounds of block
        top, bottom, left, right = get_bounds(x, y, descriptor_offset, 1)
        # Adjust the bounds
        # top, bottom, left, right = adjust_bounds(top,bottom,left,right,prev_frame.shape[0], prev_frame.shape[1])

        # Get descriptor for previous image
        prev_frame_intensities = prev_frame[top:bottom, left:right]
        prev_frame_descriptor = custom_descriptor(prev_frame_intensities)


        # Define bounds of search area
        top, bottom, left, right = get_bounds(x, y, descriptor_offset, search_offset)

        # Adjust the bounds
        # top, bottom, left, right = adjust_bounds(top,bottom,left,right, prev_frame.shape[0], prev_frame.shape[1])

        # Get search window
        search_window = curr_frame[top:bottom, left:right]

        # Compute harris corners for search window
        harris_corners = compute_harris(search_window)

        # Threshold harris corners
        harris_corner_indices = np.argwhere(harris_corners > .7 * harris_corners.max())

        # Recall numpy arrays use y,x indexing
        harris_corner_indices = np.flip(harris_corner_indices, axis=1)

        # DEBUGGING: SHOW HARRIS CORNERS
        harris_corner_indices_adjusted = np.zeros_like(harris_corner_indices)
        harris_corner_indices_adjusted[:, 0] = x - int(search_offset * descriptor_offset) + harris_corner_indices[:, 0]
        harris_corner_indices_adjusted[:, 1] = y - int(search_offset * descriptor_offset) + harris_corner_indices[:, 1]
        harris_points.extend(harris_corner_indices_adjusted.tolist())

        # Slide window throughout search area of size equal
        # to feature descriptor block
        min_sum_squares = float('inf')
        curr_desc_point_x = x
        curr_desc_point_y = y

        harris_corner_indices_adjusted = harris_corner_indices_adjusted.tolist()
        for (i, j) in harris_corner_indices_adjusted:
            top, bottom, left, right = get_bounds(i, j, descriptor_offset, 1)
            # top, bottom, left, right = adjust_bounds(top, bottom, left, right, prev_frame.shape[0], prev_frame.shape[1])
            curr_frame_intensities = curr_frame[top:bottom, left:right]
            curr_frame_descriptor = custom_descriptor(curr_frame_intensities)

            # Compute sum of squared diff
            sum_squares_tmp = np.sum((curr_frame_descriptor - prev_frame_descriptor) ** 2)
            if sum_squares_tmp < min_sum_squares:
                min_sum_squares = sum_squares_tmp
                curr_desc_point_x = i
                curr_desc_point_y = j
        curr_points.append((curr_desc_point_x, curr_desc_point_y))
    return curr_points


def compute_harris(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # result is dilated for marking the corners, not important
    harris_frame = cv2.cornerHarris(gray_frame, 5, 3, 0.04)
    harris_frame = cv2.dilate(harris_frame, None)
    return harris_frame


def draw_point(frame, x, y, color, radius):
    cv2.circle(frame, (x, y), radius, color, -1)


def draw_points(frame, points, color, radius):
    for (x, y) in points:
        draw_point(frame, x, y, color, radius)


current_frame_gui = None
clicked_points = []
harris_points = []


class Modes:
    MOVIE = 1
    IMAGE = 2


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
    x,y = points[0]
    x = x-offset
    y = y-offset
    points_offset.append([x,y])

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

    # bottom right
    x, y = points[3]
    x = x + offset
    y = y + offset
    points_offset.append([x, y])

    return points_offset

def main():
    # Open and save video files using unique path id
    id_ = 1

    ###### TRACKING VIDEO
    # Open video stream to input movie
    tracking_video = "inputs/tracking_videos/{}.MOV".format(id_)
    track_cap = cv2.VideoCapture(tracking_video)
    start_frame = 0
    # Get metadata from input movie
    track_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = track_cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(track_cap.get(3))
    frame_height = int(track_cap.get(4))

    ###### OUTPUT VIDEO
    # Define the codec and create VideoWriter object
    # to write a new movie to disk
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_composite = cv2.VideoWriter('outputs/video_output_composite_{}.avi'.format(id_), fourcc, 15,
                                    (frame_width, frame_height))
    out_tracking = cv2.VideoWriter('outputs/video_output_tracking_{}.avi'.format(id_), fourcc, 15, (frame_width, frame_height))

    ###### Composite Input
    mode = Modes.MOVIE

    # Choose to composite a video or image into the tracked planar object
    composite_cap = None
    composite_image = None

    if mode == Modes.IMAGE:
        composite_image = cv2.imread("inputs/composite_images/brick_wall.JPG")
    elif mode == Modes.MOVIE:
        composite_cap = cv2.VideoCapture("inputs/composite_videos/space.mp4")

    ####### SKIP SET (OPTIONAL)
    # Write first set of frames to disk without any image
    # warping, feature tracking
    # (this could be an introduction of some sort)
    current_frame = None
    skip_set = 0
    for i in range(skip_set):
        ret, current_frame = track_cap.read()
        if not ret:
            print("Unable to process frame...terminating script")
            return 1
        cv2.imshow('No Feature Tracking Frames', current_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Write frame to disk
        out_composite.write(current_frame)
        out_tracking.write(current_frame)
    cv2.destroyAllWindows()

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

    if len(clicked_points) < 4:
        print("You must add 4 points exactly...terminating script")
        return 1

    # Write first frame to disk
    out_composite.write(current_frame_gui)
    out_tracking.write(current_frame_gui)

    ###### FEATURE TRACKING FRAMES
    print("User added the following points {}".format(clicked_points))
    print("Starting Harris Corner Tracking...")
    prev_frame = current_frame
    prev_frame_points = clicked_points

    counter = 0
    while track_cap.isOpened():
        # Get frame from Tracking video
        ret, current_frame = track_cap.read()
        if not ret:
            print("Unable to process frame...terminating script")
            return 1

        # Get frame from composite video
        if mode == Modes.MOVIE:
            ret, composite_image = composite_cap.read()

        # add small constant to each pixel in input image to ensure that
        # the image has no pure black pixels
        # this is neccessary to utilize the input image
        # as a mask when inserting it onto a movie
        # frame
        # ENSURE THE IMAGE PIXEL INTENSITIES DO NOT OVERFLOW (UINT8)
        composite_image[composite_image != 255] += 1

        # Resize image such that it is the same as the video resolution
        composite_image = cv2.resize(composite_image, (frame_width, frame_height))

        # Compute sum of squared diff between current and prev harris images
        curr_frame_points = sum_of_square_diff(prev_frame, current_frame, prev_frame_points)

        # Apply offset to cover frame markers
        curr_frame_points_offset = apply_point_offset(curr_frame_points)

        current_frame_output = current_frame.copy()

        # print("Points {}".format(curr_frame_points))
        draw_points(current_frame_output, harris_points, (0, 0, 255), 5)
        draw_points(current_frame_output, curr_frame_points, (0, 255, 0), 3)

        # Create point correspondences for perspective transformation
        curr_frame_points_offset_array = np.array(curr_frame_points_offset).astype(np.float32)
        input_image_boundary_points_array = np.array(
            [(0, 0), (composite_image.shape[1], 0), (0, composite_image.shape[0]),
             (composite_image.shape[1], composite_image.shape[0])], dtype=np.float32)

        # Estimate perspective transformation
        M = cv2.getPerspectiveTransform(input_image_boundary_points_array, curr_frame_points_offset_array)
        maxWidth = current_frame_output.shape[1]
        maxHeight = current_frame_output.shape[0]

        # Warp composite image using perspective transformation matrix
        warped = cv2.warpPerspective(composite_image, M, (maxWidth, maxHeight))

        # use warped as mask to superimpose warped on current background
        # frame
        mask = (warped == [0, 0, 0]).all(-1)
        assert (current_frame.shape == composite_image.shape)
        current_frame_output_composite = np.where(mask[..., None], current_frame, warped)

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
        counter += 1

    cv2.destroyAllWindows()
    print("Finished Processing All Frames")
    return 0


if __name__ == "__main__":
    main()
