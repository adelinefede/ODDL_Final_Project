#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import cv2
import time
import argparse

#posture classes
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
    
    if input_shape[3] == 1 and len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to match the model's expected input dimensions
    height, width = input_shape[1], input_shape[2]
    resized_img = cv2.resize(image, (width, height))
    
    # Add batch dimension
    processed_img = np.expand_dims(resized_img, axis=0)
    
    
    if input_dtype == np.int8 or input_dtype == np.uint8:
        # For INT8 quantized model
        processed_img = processed_img.astype(np.int8)
    else:
        # For float model, normalize to [0,1]
        processed_img = processed_img.astype(np.float32) / 255.0
    
    
    if len(processed_img.shape) == 3:
        processed_img = np.expand_dims(processed_img, axis=3)
        
    return processed_img

def run_inference(interpreter, image):
    """Run inference on an image."""
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    
    interpreter.set_tensor(input_details['index'], image)
    
    
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    
    output = interpreter.get_tensor(output_details['index'])
    
    return output, inference_time

def interpret_output(output, output_details):
    """Convert raw model output to meaningful predictions."""
    
    if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
        
        scale = output_details.get('quantization_parameters', {}).get('scales', [1.0])[0]
        zero_point = output_details.get('quantization_parameters', {}).get('zero_points', [0])[0]
        
        
        dequantized = (output.astype(np.float32) - zero_point) * scale
        
        # For binary (good vs bad posture)
        if dequantized.size == 1:
            score = dequantized[0][0]
            posture_class = 0 if score > 0 else 1
            confidence = abs(score)
            return posture_class, confidence, dequantized
        
        
        else:
            posture_class = np.argmax(dequantized)
            confidence = np.max(dequantized)
            return posture_class, confidence, dequantized
    else:
        
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
    parser = argparse.ArgumentParser(description='Run headless posture detection on Raspberry Pi')
    parser.add_argument('--model', type=str, default='pruned_quantized_posture_model.tflite', 
                        help='Path to TFLite model file')
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera index (usually 0 for built-in camera)')
    parser.add_argument('--delay', type=int, default=3, 
                        help='Delay between detections in seconds (default: 3)')
    parser.add_argument('--num_runs', type=int, default=0,
                        help='Number of inference runs (0 for infinite)')
    args = parser.parse_args()
    
    
    print(f"Loading model from {args.model}...")
    interpreter = load_model(args.model)
    
    
    input_shape, input_dtype = get_input_details(interpreter)
    output_details = get_output_details(interpreter)
    
    print(f"Model loaded successfully!")
    print(f"Input shape: {input_shape}")
    print(f"Input dtype: {input_dtype}")
    print(f"Output index: {output_details['index']}")
    print(f"Output dtype: {output_details['dtype']}")
    print(f"Delay between detections: {args.delay} seconds")
    
    
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print(f"Running posture detection every {args.delay} seconds.")
    
    
    count = 0
    try:
        while args.num_runs == 0 or count < args.num_runs:
            
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            
            processed_img = preprocess_image(frame, input_shape, input_dtype)
            
            
            raw_output, inference_time = run_inference(interpreter, processed_img)
            
            
            posture_class, confidence, dequantized = interpret_output(raw_output, output_details)
            posture_label = POSTURE_CLASSES.get(posture_class, f"Unknown Class {posture_class}")
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Posture: {posture_label}")
            print(f"Raw output: {raw_output}")
            print(f"Dequantized: {dequantized}")
            print(f"Inference time: {inference_time*1000:.2f}ms")
            
            next_detection = time.localtime(time.time() + args.delay)
            next_time_str = time.strftime("%H:%M:%S", next_detection)
            print(f"Next detection in {args.delay} seconds at {next_time_str}")
            
            count += 1
            if args.num_runs == 0 or count < args.num_runs:
                time.sleep(args.delay)
                
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        print(f"Completed {count} inference runs. Exiting.")

if __name__ == "__main__":
    main() 