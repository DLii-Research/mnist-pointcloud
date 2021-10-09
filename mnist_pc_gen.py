"""
Generate MNIST digit point clouds.

Usage: python3 mnist_pc_gen <num_points> [output_path=./] [threshold=50]
"""

from os.path import join
import numpy as np
import tensorflow as tf
import sys

def generate_mnist_pointclouds(dataset, num_points=100, threshold=50):
    # Create a 1-dimensional array of pixels across all images
    img_ids, y_pixels, x_pixels = np.nonzero(dataset > threshold)
    pixels = np.column_stack((x_pixels, 28 - y_pixels))
    
    # Determine the starting pixel index of each image
    img_ids, pixel_counts = np.unique(img_ids, return_counts=True)
    pixel_index_offsets = np.roll(np.cumsum(pixel_counts), 1)
    pixel_index_offsets[0] = 0
    
    # Generate random pixel indices for each image. (len_dataset, num_points)
    random_pixel_indices = np.random.uniform(size=(num_points, dataset.shape[0]))
    pixel_indices = np.floor(pixel_counts[img_ids]*random_pixel_indices).astype(dtype=int).T
    pixel_indices += pixel_index_offsets.reshape(-1, 1)
    
    # Generate the point clouds
    points = pixels[pixel_indices].astype(float)
    points += np.random.uniform(size=points.shape)
    
    return points


def main(argv):
    
    if not (1 < len(argv) < 5):
        print("Usage: python3 mnist_pc_gen <num_points> [output_path=./] [threshold=50]")
        return 1
    
    # Commandline arguments
    num_points = int(argv[1])
    output_path = argv[2].strip() if len(argv) >= 3 else "./"
    threshold = int(argv[3]) if len(argv) >= 4 else 50
    
    # Load MNIST using Tensorflow
    (x_train_img, y_train), (x_test_img, y_test) = tf.keras.datasets.mnist.load_data()
  
    # Generate point clouds
    x_train = generate_mnist_pointclouds(x_train_img, num_points, threshold)
    x_test = generate_mnist_pointclouds(x_test_img, num_points, threshold)

    # Save the point clouds
    np.savez(join(output_path, f"mnist_point_cloud_train_{num_points}"), x_train)
    np.savez(join(output_path, f"mnist_point_cloud_test_{num_points}"), x_test)
    np.save(join(output_path, f"mnist_point_cloud_train_{num_points}_labels"), y_train)
    np.save(join(output_path, f"mnist_point_cloud_test_{num_points}_labels"), y_test)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))