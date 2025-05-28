import requests
import argparse
import time
import json

def predict_image(image_path):
    url = "http://localhost:8000/predict"
    with open(image_path, "rb") as img_file:
        files = {'file': img_file}
        response = requests.post(url, files=files)
    response.raise_for_status()
    return response.json()

def run_custom_tests():
    test_images = ["n01440764_tench.jpeg", "n01667114_mud_turtle.JPEG"]
    for img in test_images:
        start_time = time.time()
        result = predict_image(img)
        elapsed_time = time.time() - start_time
        print(f"Test image: {img}, Class ID: {result['class_id']}, Response Time: {elapsed_time:.2f}s")

    # Additional monitoring tests
    print("Running additional monitoring tests...")
    response_times = []
    for _ in range(5):
        start_time = time.time()
        predict_image(test_images[0])
        response_times.append(time.time() - start_time)
    avg_response_time = sum(response_times) / len(response_times)
    print(f"Average response time over 5 requests: {avg_response_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cerebrium deployed model.")
    parser.add_argument("--image", type=str, help="Path to the image to classify.")
    parser.add_argument("--run-tests", action="store_true", help="Run preset custom tests.")

    args = parser.parse_args()

    if args.run_tests:
        run_custom_tests()
    elif args.image:
        result = predict_image(args.image)
        print(f"Class ID: {result['class_id']}")
    else:
        print("Please provide an image path or use --run-tests flag.")