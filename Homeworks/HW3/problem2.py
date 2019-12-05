import cv2
import numpy as np

def get_bounds(x,y,descriptor_offset,search_offset):
    # For descriptor blocks, search_offset should be 1
    # This defines bounds of sliding window
    top = y - int(descriptor_offset * search_offset)
    bottom = y + int(descriptor_offset * search_offset)
    left = x - int(descriptor_offset * search_offset)
    right = x + int(descriptor_offset * search_offset)
    return top,bottom,left,right

def adjust_bounds(top,bottom,left,right,image_height,image_width):
    if top < 0:
        top = 0
    if bottom > image_height:
        bottom = image_height
    if left < 0:
        left = 0
    if right > image_width:
        right = image_width
    return top,bottom,left,right

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
        top, bottom, left, right = get_bounds(x,y,descriptor_offset,1)
        # Adjust the bounds
        # top, bottom, left, right = adjust_bounds(top,bottom,left,right,prev_frame.shape[0], prev_frame.shape[1])

        # Get descriptor for previous image
        prev_frame_descriptor = prev_frame[top:bottom,left:right]

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
        harris_corner_indices_adjusted[:,0] = x - int(search_offset * descriptor_offset) + harris_corner_indices[:,0]
        harris_corner_indices_adjusted[:,1] = y - int(search_offset * descriptor_offset) + harris_corner_indices[:,1]
        harris_points.extend(harris_corner_indices_adjusted.tolist())

        # Slide window throughout search area of size equal
        # to feature descriptor block
        min_sum_squares = float('inf')
        curr_desc_point_x = x
        curr_desc_point_y = y

        harris_corner_indices_adjusted = harris_corner_indices_adjusted.tolist()
        for (i,j) in harris_corner_indices_adjusted:
            top, bottom, left, right = get_bounds(i, j, descriptor_offset, 1)
            # top, bottom, left, right = adjust_bounds(top, bottom, left, right, prev_frame.shape[0], prev_frame.shape[1])
            curr_frame_descriptor = curr_frame[top:bottom,left:right]

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


def draw_point(frame, x, y, color,radius):
    cv2.circle(frame, (x, y), radius, color, -1)


def draw_points(frame, points,color,radius):
    for (x, y) in points:
        draw_point(frame, x, y,color,radius)


current_frame_gui = None
clicked_points = []
harris_points = []

# POINTS SHOULD BE ADDED IN THE FOLLOWING ORDER:
#
#
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        draw_point(current_frame_gui, x, y, (0,255,0),5)

def main():
    # Input input_movie Path
    id_ = 7
    input_movie = "./video_footage/tracking_video{}_input.MOV".format(id_)

    # Input Image Path (this will stick onto the bounding box initially created by a user
    # and tracked using harris corners)
    input_image = cv2.imread("./video_footage/brick_wall.JPG")
    print(input_image.shape)
    
    # add small constant to each pixel in input image to ensure that
    # the image has no pure black pixels
    # this is neccessary to utilize the input image
    # as a mask when inserting it onto a movie
    # frame
    # ENSURE THE IMAGE PIXEL INTENSITIES DO NOT OVERFLOW (UINT8)
    input_image[input_image != 255] += 1

    # Open video stream to input movie
    cap = cv2.VideoCapture(input_movie)
    start_frame = 100
    
    # Get metadata from input movie
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Resize image such that it is the same as the video resolution
    input_image = cv2.resize(input_image,(frame_width,frame_height))

    # Define the codec and create VideoWriter object
    # to write a new movie to disk
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_composite = cv2.VideoWriter('video_output_composite_{}.avi'.format(id_), fourcc, 15, (frame_width, frame_height))
    out_tracking = cv2.VideoWriter('video_output_tracking_{}.avi'.format(id_), fourcc, 15, (frame_width, frame_height))

    # Write first set of frames to disk without any image
    # warping, feature tracking
    # (this could be an introduction of some sort)
    current_frame = None
    skip_set = 0
    for i in range(skip_set):
        ret, current_frame = cap.read()
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

    # Set mouse clip
    cv2.namedWindow("Create Features To Track")
    cv2.setMouseCallback("Create Features To Track", click)

    # Get a video frame and select 4 points
    # From here on out all frames will be tracked
    ret, current_frame = cap.read()
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

    print("User added the following points {}".format(clicked_points))
    print("Starting Harris Corner Tracking...")
    prev_frame = current_frame
    prev_frame_points = clicked_points

    counter = 0
    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            print("Unable to process frame...terminating script")
            return 1

        # Compute sum of squared diff between current and prev harris images
        curr_frame_points = sum_of_square_diff(prev_frame, current_frame, prev_frame_points)
        current_frame_output = current_frame.copy()
        
        # print("Points {}".format(curr_frame_points))
        draw_points(current_frame_output, harris_points,(0,0,255),5)
        draw_points(current_frame_output, curr_frame_points,(0,255,0),3)

        # Create point correspondences for perspective transformation
        curr_frame_points_array = np.array(curr_frame_points).astype(np.float32)
        input_image_boundary_points_array = np.array([(0,0),(input_image.shape[1],0),(0,input_image.shape[0]),(input_image.shape[1],input_image.shape[0])],dtype=np.float32)

        # Estimate perspective transformation
        M = cv2.getPerspectiveTransform(input_image_boundary_points_array, curr_frame_points_array)
        maxWidth = current_frame_output.shape[1]
        maxHeight = current_frame_output.shape[0]

        # Warp composite image using perspective transformation matrix
        warped = cv2.warpPerspective(input_image, M, (maxWidth, maxHeight))

        # use warped as mask to superimpose warped on current background
        # frame
        mask = (warped == [0, 0, 0]).all(-1)
        assert(current_frame.shape == input_image.shape)
        current_frame_output_composite = np.where(mask[...,None],current_frame,warped)

        #Display the frame for diagnostic purposes
        cv2.imshow('Final Frame', current_frame_output)
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
