from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer('human')
sample_images = ["data/training/10/11666.0.jpg", "data/training/10/11666.1.jpg"]  # Replace with actual image paths
visualized_batch = list(inferencer(sample_images))

for vis in visualized_batch:
    print(vis)
