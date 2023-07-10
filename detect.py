import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np


import RPi.GPIO as GPIO

# Set the GPIO mode
GPIO.setmode(GPIO.BOARD)

# Define the GPIO pin connected to the servo motor
servo_pin = 18

# Set up the GPIO pin as an output
GPIO.setup(servo_pin, GPIO.OUT)

# Create a PWM object for the servo motor
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz (20 ms period)

# Set the initial position (e.g., 90 degrees)
initial_position = 90
pwm.start(initial_position/18 + 2.5)


#for WSL
#def load_labels(path='/mnt/d/debian/object-detection-Raspberry-Pi/labels.txt'):
#for RPi 
def load_labels(path='/home/pi/Desktop/angle/Human-Detection-Pi/labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results



def map_value(value, from_min, from_max, to_min, to_max):
    # Maps a value from one range to another
    from_span = from_max - from_min
    to_span = to_max - to_min
    value_scaled = float(value - from_min) / float(from_span)
    return to_min + (value_scaled * to_span)


def main():
    labels = load_labels()
    interpreter = Interpreter('human.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(0)
    CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        res = detect_objects(interpreter, img, 0.2)
        print(res)

        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            bbox_center_x = xmin + bbox_width / 2
            bbox_center_y = ymin + bbox_height / 2

            print("\n")

            print("Bounding Box Center: ({}, {})".format(bbox_center_x, bbox_center_y))

            print("\n")
            

            # Calculate the desired position based on the bounding box center
            servo_range = 180  # The range of motion of your servo motor
            center_x = bbox_center_x  # The center X-coordinate from the bounding box
            center_y = bbox_center_y  # The center Y-coordinate from the bounding box
            reversed_x = CAMERA_WIDTH - center_x
            reversed_y = CAMERA_HEIGHT - center_y
            
            # Map the reversed center X-coordinate to the servo motor's range
            mapped_x = map_value(reversed_x, 0, CAMERA_WIDTH, 0, servo_range)

            print("mapped_x",mapped_x,"\n")

            # Map the center Y-coordinate to the servo motor's range (if needed)
            mapped_y = map_value(reversed_y, 0, CAMERA_HEIGHT, 0, servo_range)

            # Update the servo motor position
            pwm.ChangeDutyCycle(mapped_x/18 + 2.5)

            # If needed, update the Y-axis position as well
            # pwm.ChangeDutyCycle(mapped_y/18 + 2.5)
            
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 

        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
