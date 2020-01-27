from utilities import *
import poisson_algorithm as pa
import cv2
import os
import sys

# Enum
Image, Video = range(1, 3)


def main():
    # Read images
    input_dir_base = './inputs'
    output_dir_base = './outputs'

    # Determine poisson_mode: Can do either inpainting or compositing
    mode = None
    for subdir in list(os.walk(input_dir_base))[0][1]:
        if (subdir.lower().__contains__("inpaint")):
            mode = 'Inpaint'
        elif (subdir.lower().__contains__("composite")):
            mode = 'Composite'
        else:
            sys.exit('Each input directory should contain one of the following terms: inpaint or composite')
        print("Mode: {} Input | Directory: {}".format(mode, subdir))

        # Create output directory
        output_dir = f"{output_dir_base}/{subdir}"
        make_dir(output_dir)

        # Extract paths to source, mask, and target files
        # Return codes
        # 1 file is an image, 2 file is a movie, 3 no file found that matches criteria
        input_dir = f"{input_dir_base}/{subdir}"
        retS, S_path = get_file_path(input_dir, "source")
        retT, T_path = get_file_path(input_dir, "target")
        retM, M_path = get_file_path(input_dir, "mask")
        S = cv2.imread(S_path)
        M = cv2.imread(M_path, 0)  # Grayscale
        # Target (will be defined later)
        T = None

        # Used when composite target is a movie; only applies to compositing
        cap_target = None
        composite_writer = None
        mixed_composite_writer = None

        # This means the composite target is an image
        # instead of an image
        if (mode == 'Composite') and (retT == Image):
            T = cv2.imread(T_path)
        # This means the composite target is actually a movie
        # instead of an image
        elif (mode == 'Composite') and (retT == Video):
            cap_target = cv2.VideoCapture(T_path)
            start_frame_number = 0
            cap_target.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            fps = cap_target.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap_target.get(3))
            frame_height = int(cap_target.get(4))
            composite_writer = cv2.VideoWriter(f"{output_dir}/regular_composite.avi", cv2.VideoWriter_fourcc(*"MJPG"),
                                               fps, (frame_width, frame_height))
            mixed_composite_writer = cv2.VideoWriter(f"{output_dir}/mixed_composite.avi",
                                                     cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))
            ret, T = cap_target.read()

        ## Sanity check: ensure all files exist (Target not required for inpainting)
        if ((mode == 'Composite') and (T is None)) or (S is None) or (M is None):  # ensure all files exist
            print("Error reading at least one of the input files")
            continue
        ## Sanity check: ensure source and target have same dimensions (only true for composite poisson_mode)
        if (mode == 'Composite') and (S.shape[:2] != T.shape[:2]):
            print("Source and Target dimensions should be the same (for composite poisson_mode)")
            continue
        ## Sanity check: ensure source and mask have same dimensions
        if S.shape[:2] != M.shape[:2]:
            print("Source and Mask dimensions should be the same")
            continue

        if mode == 'Inpaint':
            print("Original Image Dimensions: S {} M {}".format(S.shape, M.shape))
        elif mode == 'Composite':
            print("Original Image Dimensions: S {} T {} M {}".format(S.shape, T.shape, M.shape))

        if retT == Image or retS == Image:
            fx = 0.25
            fy = 0.25
            if mode == 'Composite':
                S = cv2.resize(S, (0, 0), fx=fx, fy=fy)
                T = cv2.resize(T, (0, 0), fx=fx, fy=fy)
                M = cv2.resize(M, (0, 0), fx=fx, fy=fy)
                print("Resized Image Dimensions: S {} T {} M {}".format(S.shape, T.shape, M.shape))
                source_intensities_3ch, source_coordinates = pa.solve_possion(S, T, M, output_dir, poisson_mode="Regular")
                source_intensities_mg_3ch, _ = pa.solve_possion(S, T, M, output_dir, poisson_mode="Mixed_Gradients")
                composite_image = pa.blend(source_intensities_3ch, T, source_coordinates)
                composite_image_mixed = pa.blend(source_intensities_mg_3ch, T, source_coordinates)
                cv2.imwrite(f"{output_dir}/composite_image_regular.png", composite_image)
                cv2.imwrite(f"{output_dir}/composite_image_mixed_gradient.png", composite_image_mixed)
            elif mode == "Inpaint":
                S = cv2.resize(S, (0, 0), fx=fx, fy=fy)
                M = cv2.resize(M, (0, 0), fx=fx, fy=fy)
                print("Resized Image Dimensions: S {} M {}".format(S.shape, M.shape))
                source_intensities_3ch, source_coordinates = pa.solve_possion(S, T, M, output_dir, poisson_mode="Inpaint")
                inpainting_image = pa.blend(source_intensities_3ch, S, source_coordinates)
                cv2.imwrite(f"{output_dir}/inpainting_image.png", inpainting_image)
        elif retT == Video:
            # Poisson Intensities
            source_intensities_3ch = None
            # Mixed Gradient Poisson Intensities
            source_intensities_mg_3ch = None
            # Source pixel coordinates
            source_coordinates = None
            count = 0

            while cap_target.isOpened():
                ret, T = cap_target.read()
                if T is None:
                    break
                if count == 0:
                    source_intensities_3ch, source_coordinates = pa.solve_possion(S, T, M, output_dir,
                                                                                  poisson_mode="Regular")
                    source_intensities_mg_3ch, _ = pa.solve_possion(S, T, M, output_dir, poisson_mode="Mixed_Gradients")

                composite_image = pa.blend(source_intensities_3ch, T, source_coordinates)
                composite_image_mixed = pa.blend(source_intensities_mg_3ch, T, source_coordinates)

                cv2.imshow('Composite Regular', composite_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                composite_writer.write(composite_image)
                mixed_composite_writer.write(composite_image_mixed)

                count += 1
        if retT == Video:
            cap_target.release()
            composite_writer.release()
            mixed_composite_writer.release()

        print("")

if __name__ == '__main__':
    main()
