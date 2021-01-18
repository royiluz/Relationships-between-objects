from imageai.Detection import ObjectDetection
import tensorflow as tf

if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()
    detector = ObjectDetection()

    model_path = "./models/yolo.h5"
    # input_path = "./input/pic_11.jpg"
    input_path = "./input/pic_22.jpeg"
    output_path = "./new_pic.jpg"

    # detector.setModelTypeAsTinyYOLOv3()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()
    detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

    for eachItem in detection:
        print(eachItem["name"], " : ", eachItem["percentage_probability"])
