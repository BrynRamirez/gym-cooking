import cv2
import os
from itertools import product

def create_video_from_images(image_folder, output_file, fps=24):
    """Creates a video from a folder of images.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_file (str): Path to the output video file.
        fps (int): Frames per second for the video.

    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure images are in the correct order

    if not images:
        print("No images found in the folder.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()

# Upload each file in record into mp4 format into recordings
input_location = "C:/Users/bryng/gym-cooking/gym_cooking/misc/game/record"
models = ["bd", "up", "dc", "fb", "greedy"]
output_location = "C:/Users/bryng/gym-cooking/gym_cooking/misc/recordings"

#image_folder = "C:/Users/bryng/gym-cooking/gym_cooking/misc/game/record/open-divider_salad_agents2_seed1_model1-bd_model2-bd"
#output_file = "C:/Users/bryng/gym-cooking/gym_cooking/misc/recordings/open-divider_salad_agents2_seed1_model1-bd_model2-bd.mp4"

for model1, model2 in product(models, repeat=2):
    mname = f"open-divider_salad_agents2_seed1_model1-{model1}_model2-{model2}"
    input_full_path = os.path.join(input_location, mname)

    output_name =f"open-divider_salad_agents2_seed1_model1-{model1}_model2-{model2}.mp4"
    output_full_path = os.path.join(output_location, output_name)

    create_video_from_images(input_full_path, output_full_path, 10)
