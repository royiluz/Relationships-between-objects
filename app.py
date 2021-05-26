from imageai.Detection import ObjectDetection
import tensorflow as tf
from flask import Flask, render_template, request


def getRelations(img_path):

    model_path = "./models/yolo.h5"
    output_path = "./new_pic.jpg"

    # Constants
    TABLE = "dining table"
    CHAIR = "chair"
    TABLE_CHAIR = [TABLE, CHAIR]

    # Create detector with yolo
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()
    detection = detector.detectObjectsFromImage(input_image=img_path, output_image_path=output_path)

    relations = []

    # Iterate each pair of objects
    for i in range(len(detection)):
        for j in range(i + 1, len(detection)):
            obj_1 = detection[i]
            obj_2 = detection[j]

            # Make sure that CHAIR/TABLE is obj_1:
            if obj_2["name"] in TABLE_CHAIR:
                temp = obj_1
                obj_1 = obj_2
                obj_2 = temp

            obj_1_x1, obj_1_y1, obj_1_x2, obj_1_y2 = obj_1["box_points"]
            obj_2_x1, obj_2_y1, obj_2_x2, obj_2_y2 = obj_2["box_points"]
            obj_1_width = obj_1_x2 - obj_1_x1
            obj_2_width = obj_2_x2 - obj_2_x1

            # Check relation between Cup/Glass and Chair:
            if (obj_1["name"] == CHAIR) and (obj_2["name"] not in TABLE_CHAIR):
                chair_height = obj_1_y2 - obj_1_y1
                chair_width = obj_1_width
                # If cup between the chair x borders:
                if obj_1_x1 - chair_width * 0.2 < obj_2_x1 and obj_2_x2 < obj_1_x2 + chair_width * 0.2:
                    if obj_1_y1 < obj_2_y1:
                        if obj_1_y1 + chair_height * 0.66 < obj_2_y2:
                            print(f'{obj_2["name"]} under chair')
                            relations.append(f'{obj_2["name"]} under chair')
                        else:
                            print(f'{obj_2["name"]} above chair')
                            relations.append(f'{obj_2["name"]} above chair')

            # Check relation between Cup and Table:
            elif (obj_1["name"] == TABLE) and (obj_2["name"] not in TABLE_CHAIR):
                table_height = obj_1_y2 - obj_1_y1
                if obj_1_x1 < obj_2_x1 and obj_2_x2 < obj_1_x2:
                    if (obj_2_y2 < obj_1_y1 + table_height * 0.1) and (obj_1_y1 - table_height * 0.2 < obj_2_y2):
                        print(f'{obj_2["name"]} above table')
                        relations.append(f'{obj_2["name"]} above table')
                    elif obj_2_y2 < obj_2_y2:
                        print(f'{obj_2["name"]} under table')
                        relations.append(f'{obj_2["name"]} under table')

            # Check relation between any objects Left and Right:
            else:
                if (obj_1_x2 - obj_1_width * 0.5 < obj_2_x1) or (obj_2_x1 + obj_2_width * 0.5 > obj_1_x2):
                    print(f"{obj_1['name']} left to {obj_2['name']}")
                    relations.append(f"{obj_1['name']} left to {obj_2['name']}")
                elif (obj_1_x1 + obj_1_width * 0.5 > obj_2_x2) or (obj_2_x2 - obj_2_width * 0.5 < obj_1_x1):
                    print(f"{obj_1['name']} right to {obj_2['name']}")
                    relations.append(f"{obj_1['name']} right to {obj_2['name']}")
    return relations


if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()

    # input/output values:
    input_path = "./input/2.jpg"
    # relation_array = getRelations(input_path)
    # print(relation_array)

    # Using flask
    app = Flask(__name__)

    @app.route('/')
    def show_predict_stock_form():
        return render_template('predictorform.html')

    @app.route('/results', methods=['POST'])
    def upload_file():
        uploaded_file = request.files['image_file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
            relations = getRelations(uploaded_file.filename)
            print(relations)
        return render_template('resultsform.html', relations=relations)

    app.run("localhost", "9999", debug=True)








