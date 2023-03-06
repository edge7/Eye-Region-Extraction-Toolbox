import cv2
import mediapipe as mp

# Initialize face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,  # Set to False for video streams
    max_num_faces=1,
    refine_landmarks=False,  # Additional landmarks not needed
    min_detection_confidence=0.5)

# Open camera capture
cap = cv2.VideoCapture(4)


# Function to invert normalization of landmarks
def invert_normalization(x, y, w, h):
    return int(x * w), int(y * h)


def fix_aspect_ratio(image, y, y1, required_ratio):
    new_h = int(image.shape[1] * required_ratio)
    diff_h = int((new_h - image.shape[0]) / 2)
    return y - diff_h, y1 + diff_h


def get_aspect_ratio(region):
    region_width = region.shape[1]
    region_height = region.shape[0]
    region_aspect_ratio = float(region_height) / float(region_width)
    print("aspect ratio", region_aspect_ratio)
    return region_aspect_ratio


while cap.isOpened():
    success, image = cap.read()  # Read from camera capture
    if not success or image is None or image.shape[0] == 0:
        print("Ignoring empty camera frame.")
        continue

    # Convert BGR image to RGB
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image with face mesh
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        print("unable to extract face")
        continue

    # Extract eye region from image
    to_check = results.multi_face_landmarks[0].landmark
    image_width = image.shape[1]
    image_height = image.shape[0]
    up_left_x, up_left_y = invert_normalization(x=to_check[71].x, y=to_check[71].y, w=image_width, h=image_height)
    down_left_x, down_left_y = invert_normalization(x=to_check[71].x, y=to_check[123].y, w=image_width, h=image_height)
    up_right_x, up_right_y = invert_normalization(x=to_check[301].x, y=to_check[71].y, w=image_width, h=image_height)
    down_right_x, down_right_y = invert_normalization(x=to_check[301].x, y=to_check[123].y, w=image_width,
                                                      h=image_height)
    width_eyes = (down_right_x - down_left_x)
    center_x = int(width_eyes / 2)
    safe_increase = int(width_eyes * .2)
    down_right_x = int(down_right_x + safe_increase)
    down_left_x = int(down_left_x - safe_increase)
    eye_region = image[up_left_y: down_left_y, down_left_x:down_right_x, :]

    eye_region_aspect_ratio = get_aspect_ratio(eye_region)
    RATIO_THRES = 0.5
    if eye_region_aspect_ratio <= RATIO_THRES:
        up_left_y, down_left_y = fix_aspect_ratio(eye_region, up_left_y, down_left_y,  RATIO_THRES)
        eye_region = image[up_left_y: down_left_y, down_left_x:down_right_x, :]
        get_aspect_ratio(eye_region)
        print(eye_region.shape)

    res = cv2.resize(cv2.cvtColor(eye_region, cv2.COLOR_RGB2BGR), (64, 64))
    # Display images
    cv2.imshow('Eye regions', cv2.resize(res, (eye_region.shape[1], eye_region.shape[0])))
    cv2.imshow('Eye regions original', cv2.cvtColor(eye_region, cv2.COLOR_RGB2BGR))
    cv2.imshow('Full image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Exit on ESC key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release camera capture and close windows
cap.release()
cv2.destroyAllWindows()
