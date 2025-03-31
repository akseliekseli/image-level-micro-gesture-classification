from utils.VitPoseKeypoints import generate_keypoints 




if __name__ == '__main__':
    data_dir = "data/training"
    output_file = "keypoints.json"
    generate_keypoints(data_dir, output_file)

