import tf_keras as keras
import tensorflow_hub as hub
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Int8MultiArray
from sensor_msgs.msg import Image 

from cv_bridge import CvBridge, CvBridgeError

from msg import Prediction
import cv2
import numpy as np

# Create a function to load a trained model
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  new_model = keras.models.load_model(model_path,
                                      custom_objects={"KerasLayer":hub.KerasLayer})
  return new_model


class TrafficSignClassifier(Node):

    def __init__(self):
        super().__init__('traffic_sign_classifier')

        # create and load the model
        self.traffic_sign_model = load_model('Rc-Car/src/model/20260617-11321781695951-all-images-mobilenetv2-Adam.h5')
        self.labels = [
            "stop_sign",
            "30_kph",
            "50_kph",
            "80_kph",
            "130_kph",
        ]

        self.bridge = CvBridge()

        self.image_reader = self.create_subscription(Image, "/ZEDcam/image_raw", self.classification_callback, 10)

        self.traffic_sign_predictor = self.create_publisher(Prediction, '/predicted_traffic_signs', 10)
        

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis = 0)

        return image 


    def classification_callback(self, msg):

        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Preprocess
            input_tensor = self.preprocess(cv_image)

            # Predict
            predictions = self.traffic_sign_model.predict(input_tensor, verbose=0)
            class_idx = int(np.max(predictions))
            confidence = float(predictions[0][class_idx])

            label = self.labels[class_idx]

            out_msg = Prediction()
            out_msg.label = label
            out_msg.confidence = confidence
            self.traffic_sign_predictor.publish(out_msg)

            print("Predictions: label = ", label, " confidence = ", confidence)

        except CvBridgeError as e: 
            print(e) #if there is an error it will be printed out


def main(args=None):
    rclpy.init(args=args)

    traffic_sign_classifier = TrafficSignClassifier()

    try:
        rclpy.spin(traffic_sign_classifier)
    except KeyboardInterrupt:
        pass

    traffic_sign_classifier.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()