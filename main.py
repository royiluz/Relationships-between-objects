from imageai.Detection import ObjectDetection
import tensorflow as tf

if __name__ == "__main__":

    # Constants
    TABLE = "dining table"
    CUP = ["cup", "wine glass"]
    CHAIR = "chair"

    tf.compat.v1.disable_eager_execution()

    # input/output values:
    model_path = "./models/yolo.h5"
    input_path = "./input/6.jpeg"
    output_path = "./new_pic.jpg"

    # Create detector with yolo
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()
    detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

    # Iterate each pair of objects
    for obj_1 in detection:
        for obj_2 in detection:
            obj_1_x1, obj_1_y1, obj_1_x2, obj_1_y2 = obj_1["box_points"]
            obj_2_x1, obj_2_y1, obj_2_x2, obj_2_y2 = obj_2["box_points"]

            # Check relation between Table and Chair:
            if obj_1["name"] == TABLE and obj_2["name"] == CHAIR:
                chair_width = obj_2_x2 - obj_2_x1
                if obj_2_x1 + chair_width * 0.2 < obj_1_x1:
                    print("Chair left to table")
                elif obj_1_x2 < obj_2_x2 - chair_width * 0.2:
                    print("Chair right to table")

            # Check relation between Cup/Glass and Chair:
            if (obj_1["name"] in CUP) and obj_2["name"] == CHAIR:
                chair_height = obj_2_y2 - obj_2_y1
                chair_width = obj_2_x2 - obj_2_x1
                # If cup between the chair x borders:
                if obj_2_x1 - chair_width * 0.2 < obj_1_x1 and obj_1_x2 < obj_2_x2 + chair_width * 0.2:
                    if obj_2_y1 < obj_1_y1:
                        if obj_2_y1 + chair_height * 0.66 < obj_1_y2:
                            print(f'{obj_1["name"]} under chair')
                        else:
                            print(f'{obj_1["name"]} above chair')

            # Check relation between Cup and Table:
            if (obj_1["name"] == TABLE) and (obj_2["name"] in CUP):
                table_height = obj_1_y2 - obj_1_y1
                if obj_1_x1 < obj_2_x1 and obj_2_x2 < obj_1_x2:
                    if (obj_2_y2 < obj_1_y1 + table_height * 0.1) and (obj_1_y1 - table_height * 0.2 < obj_2_y2):
                        print(f'{obj_2["name"]} above table')
                    elif obj_2_y2 < obj_2_y2:
                        print(f'{obj_2["name"]} under table')
