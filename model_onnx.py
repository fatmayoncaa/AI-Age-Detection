import onnxruntime as ort
import numpy as np
import cv2
import random

class Model:
    def __init__(self, id2label, model_path, config):
        self.id2label = id2label
        self.model_path = model_path
        self.config = config
        self.image_path = None  

    # Helper function to convert label to a representative age
    def label_to_age(self, label):
        age_group = self.id2label[label]
        if '-' in age_group:
            start, end = map(int, age_group.split('-'))
            return random.randint(start,end)
        else:
            return int(age_group)

    def get_image(self, image_path):
        # Load the image
        org_image = cv2.imread(image_path)
        return org_image

    def preprocess_image(self, image_path):
        self.image_path = image_path  # Set the image_path attribute
        image = self.get_image(self.image_path)

        # Resize the image
        if self.config["do_resize"]:
            size = (self.config["size"]["width"], self.config["size"]["height"])
            image = cv2.resize(image, size, interpolation=self.config["resample"])

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Rescale the image
        if self.config["do_rescale"]:
            image = image * self.config["rescale_factor"]

        # Normalize the image
        if self.config["do_normalize"]:
            image_mean = np.array(self.config["image_mean"])
            image_std = np.array(self.config["image_std"])
            image = (image - image_mean) / image_std

        # Convert the image to float32
        image = image.astype(np.float32)

        # Transpose the image to match the input shape 
        image = np.transpose(image, (2, 0, 1))

        # Add a batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def predict_age(self, image_path):
        # Preprocess the image
        input_image = self.preprocess_image(image_path)

        # Load the ONNX model
        session = ort.InferenceSession(self.model_path)

        # Get the name of the input and output layers
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run the model to get the predictions
        predictions = session.run([output_name], {input_name: input_image})

        # Post-process the model output
        output = predictions[0]

        # Assuming the output is class probabilities
        predicted_label = np.argmax(output)
        predicted_age = self.label_to_age(predicted_label)

        return predicted_age



