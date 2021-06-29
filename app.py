import cv2
from imageai.Detection import ObjectDetection
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import numpy as np
from keras.models import load_model


def getRelations(img_path):
    # Constants
    TABLE = "dining table"
    CHAIR = "chair"
    TABLE_CHAIR = [TABLE, CHAIR]

    # Load Yolo
    model_path = "./models/yolo.h5"
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()

    output_path = "./new_pic.jpg"
    detection = detector.detectObjectsFromImage(input_image=img_path, output_image_path=output_path)
    relations = []
    names = {}
    # Iterate each pair of objects
    for i in range(len(detection)):
        if detection[i]["name"] in names:
            names[detection[i]["name"]] += 1
        else:
            names[detection[i]["name"]] = 1

        for j in range(i + 1, len(detection)):
            flag = 0
            obj_1 = detection[i]
            obj_2 = detection[j]

            # Make sure chair is obj_1 and table is obj_2 (if exist!):
            if obj_2["name"] == CHAIR and obj_1["name"] == TABLE:
                temp = obj_1
                obj_1 = obj_2
                obj_2 = temp

            # Make sure that CHAIR/TABLE is obj_1:
            elif obj_2["name"] in TABLE_CHAIR:
                temp = obj_1
                obj_1 = obj_2
                obj_2 = temp

            obj_1_x1, obj_1_y1, obj_1_x2, obj_1_y2 = obj_1["box_points"]
            obj_2_x1, obj_2_y1, obj_2_x2, obj_2_y2 = obj_2["box_points"]
            obj_1_width = obj_1_x2 - obj_1_x1
            obj_2_width = obj_2_x2 - obj_2_x1

            # Check relation between Objects and Chair:
            if (obj_1["name"] == CHAIR) and (obj_2["name"] not in TABLE_CHAIR):
                chair_height = obj_1_y2 - obj_1_y1
                chair_width = obj_1_width
                # If Object between the chair's x borders:
                if obj_1_x1 - chair_width * 0.2 < obj_2_x1 and obj_2_x2 < obj_1_x2 + chair_width * 0.2:
                    if obj_1_y1 < obj_2_y1:
                        if obj_1_y1 + chair_height * 0.66 < obj_2_y2:
                            print(f'{obj_2["name"]} under chair')
                            relations.append(f'{obj_2["name"]} under chair')
                            flag = 1
                        else:
                            print(f'{obj_2["name"]} above chair')
                            relations.append(f'{obj_2["name"]} above chair')
                            flag = 1

            # Check relation between Cup and Table:
            elif (obj_1["name"] == TABLE) and (obj_2["name"] not in TABLE_CHAIR):
                table_height = obj_1_y2 - obj_1_y1
                if obj_1_x1 < obj_2_x1 + obj_2_width * 0.2 and obj_2_x2 - obj_2_width * 0.2 < obj_1_x2:
                    if (obj_2_y2 < obj_1_y1 + table_height * 0.3) and (obj_1_y1 - table_height * 0.3 < obj_2_y2):
                        print(f'{obj_2["name"]} above table')
                        relations.append(f'{obj_2["name"]} above table')
                        flag = 1
                    elif obj_2_y2 < obj_2_y2:
                        print(f'{obj_2["name"]} under table')
                        relations.append(f'{obj_2["name"]} under table')
                        flag = 1

            # Check relation between any objects Left and Right:
            if flag == 0:
                if ((obj_1_x2 - obj_1_width * 0.5 < obj_2_x1) or (
                        obj_2_x1 + obj_2_width * 0.5 > obj_1_x2)) and obj_1_x1 < obj_2_x1:
                    print(f"{obj_1['name']} left to {obj_2['name']}")
                    relations.append(f"{obj_1['name']} left to {obj_2['name']}")
                elif ((obj_1_x1 + obj_1_width * 0.5 > obj_2_x2) or (
                        obj_2_x2 - obj_2_width * 0.5 < obj_1_x1)) and obj_1_x2 > obj_2_x2:
                    print(f"{obj_1['name']} right to {obj_2['name']}")
                    relations.append(f"{obj_1['name']} right to {obj_2['name']}")
                elif ((obj_1["name"] == CHAIR) and (obj_2["name"] == TABLE)) or (
                        (obj_2["name"] == CHAIR) and (obj_1["name"] == TABLE)):
                    print(f"CHAIR AND TABLE relations")
                    relations.append(getRelationByModel(obj_1, obj_2, img_path))

    return relations, names


def getRelationByModel(obj_1, obj_2, img_path):
    # Get Coordinates:
    obj_1_x1, obj_1_y1, obj_1_x2, obj_1_y2 = obj_1["box_points"]
    obj_2_x1, obj_2_y1, obj_2_x2, obj_2_y2 = obj_2["box_points"]

    img = cv2.imread(img_path)
    x1 = min(obj_1_x1, obj_2_x1)
    y1 = min(obj_1_y1, obj_2_y1)
    x2 = max(obj_1_x2, obj_2_x2)
    y2 = max(obj_1_y2, obj_2_y2)

    crop_img = img[y1:y2, x1:x2]
    crop_img = cv2.resize(crop_img, (300, 300))
    crop_img = np.reshape(crop_img, [1, 300, 300, 3])

    model = load_model('ChairInFrontBehindTable.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    c = model.predict(crop_img)
    classes = model.predict_classes(crop_img)
    classes_proba = model.predict_proba(crop_img)

    print("C:" + str(c))
    print("Classes:" + str(classes))
    print(classes_proba)

    if classes[0] == 0:
        return 'Chair behind table'
    else:
        return 'Chair in front of table'


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

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
            relations, names = getRelations(uploaded_file.filename)
            im = Image.open("new_pic.jpg")
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            return render_template('resultsform.html', relations=relations, names=names,
                                   image=encoded_img_data.decode('utf-8'))
        else:
            return '', 204  # Do nothing


    app.run("localhost", "9999", debug=True)
