#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import cv2
import time
import argparse

# Define posture classes
POSTURE_CLASSES = {
    0: "Good Posture",
    1: "Bad Posture - Forward Head",
    2: "Bad Posture - Slouching",
    3: "Bad Posture - Other"
}

def load_model(model_path):
    """Load the TFLite model and allocate tensors."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    """Get model input details."""
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details['shape']
    input_dtype = input_details['dtype']
    return input_shape, input_dtype

def get_output_details(interpreter):
    """Get model output details."""
    output_details = interpreter.get_output_details()[0]
    return output_details

def preprocess_image(image, input_shape, input_dtype=None):
    """Preprocess the image for model input."""
    # Convert to grayscale if the input shape expects single channel
    if input_shape[3] == 1 and len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to match the model's expected input dimensions
    height, width = input_shape[1], input_shape[2]
    resized_img = cv2.resize(image, (width, height))
    
    # Add batch dimension
    processed_img = np.expand_dims(resized_img, axis=0)
    
    # Handle quantized models (INT8)
    if input_dtype == np.int8 or input_dtype == np.uint8:
        # For INT8 quantized model
        processed_img = processed_img.astype(np.int8)
    else:
        # For float model, normalize to [0,1]
        processed_img = processed_img.astype(np.float32) / 255.0
    
    # Add channel dimension if needed (for grayscale)
    if len(processed_img.shape) == 3:
        processed_img = np.expand_dims(processed_img, axis=3)
        
    return processed_img

def run_inference(interpreter, image):
    """Run inference on an image."""
    # Get input and output tensors
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Set the input tensor
    interpreter.set_tensor(input_details['index'], image)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Get the output tensor
    output = interpreter.get_tensor(output_details['index'])
    
    return output, inference_time

def interpret_output(output, output_details):
    """Convert raw model output to meaningful predictions."""
    # For quantized model, dequantize the output
    if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
        # Get scale and zero point for dequantization
        scale = output_details.get('quantization_parameters', {}).get('scales', [1.0])[0]
        zero_point = output_details.get('quantization_parameters', {}).get('zero_points', [0])[0]
        
        # Dequantize
        dequantized = (output.astype(np.float32) - zero_point) * scale
        
        # For binary classification (good vs bad posture)
        if dequantized.size == 1:
            score = dequantized[0][0]
            posture_class = 0 if score > 0 else 1
            confidence = abs(score)
            return posture_class, confidence, dequantized
        
        # For multi-class classification
        else:
            posture_class = np.argmax(dequantized)
            confidence = np.max(dequantized)
            return posture_class, confidence, dequantized
    else:
        # For float model
        if output.size == 1:
            score = output[0][0]
            posture_class = 0 if score > 0.5 else 1
            confidence = abs(score - 0.5) * 2
            return posture_class, confidence, output
        else:
            posture_class = np.argmax(output)
            confidence = np.max(output)
            return posture_class, confidence, output

def main():
    parser = argparse.ArgumentParser(description='Run posture detection TFLite model on Raspberry Pi')
    parser.add_argument('--model', type=str, default='pruned_quantized_posture_model.tflite', 
                        help='Path to TFLite model file')
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera index (usually 0 for built-in camera)')
    parser.add_argument('--delay', type=int, default=3, 
                        help='Delay between detections in seconds (default: 3)')
    args = parser.parse_args()
    
    # Load the TFLite model
    print(f"Loading model from {args.model}...")
    interpreter = load_model(args.model)
    
    # Get model details
    input_shape, input_dtype = get_input_details(interpreter)
    output_details = get_output_details(interpreter)
    
    print(f"Model loaded successfully!")
    print(f"Input shape: {input_shape}")
    print(f"Input dtype: {input_dtype}")
    print(f"Output index: {output_details['index']}")
    print(f"Output dtype: {output_details['dtype']}")
    print(f"Delay between detections: {args.delay} seconds")
    
    # Open the camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    
    next_detection_time = time.time()
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            current_time = time.time()
            # Check if it's time for detection
            if current_time >= next_detection_time:
                # Preprocess the image
                processed_img = preprocess_image(frame, input_shape, input_dtype)
                
                # Run inference
                raw_output, inference_time = run_inference(interpreter, processed_img)
                
                # Interpret the output
                posture_class, confidence, dequantized = interpret_output(raw_output, output_details)
                posture_label = POSTURE_CLASSES.get(posture_class, f"Unknown Class {posture_class}")
                
                # Print results
                print(f"Prediction: {posture_label}")
                print(f"Raw output: {raw_output}")
                print(f"Dequantized: {dequantized}")
                print(f"Inference time: {inference_time*1000:.2f}ms")
                
                # Schedule next detection
                next_detection_time = current_time + args.delay
                print(f"Next detection in {args.delay} seconds at {time.strftime('%H:%M:%S', time.localtime(next_detection_time))}")
            
            # Display current status on frame
            time_to_next = max(0, int(next_detection_time - current_time))
            status_text = f"Posture: {posture_label if 'posture_label' in locals() else 'Waiting for first detection...'}"
            cv2.putText(frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Next detection in: {time_to_next}s", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Posture Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 