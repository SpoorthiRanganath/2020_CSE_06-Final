from flask import Flask, request, jsonify, send_file
from serve import *
import os
import base64
from flask_cors import CORS, cross_origin

yolo = Yolov3()

IMAGE_TEMP_FOLDER = os.path.join(os.getenv("PWD"), "data")
IMAGE_RESULT_PATH = os.path.join(os.getenv("PWD"), "results")

app = Flask("__main__")
CORS(app)


@app.route("/api/infer", methods=['POST'])
def infer():

    files_stream = request.files
    if not 'image' in request.files:
        return jsonify({
            "success": False,
            "result": "file not found..."
        })

    f = request.files['image']
    f_name = f.filename

    fpath = os.path.join(IMAGE_TEMP_FOLDER, f_name)
    f.save(fpath)

    yolo.predict(fpath)

    yolo.clear()

    with open(fpath, 'rb') as image:
        imbase64 = base64.b64encode(image.read())

    return imbase64


@app.route("/api/raw_infer", methods=['POST'])
def raw_infer():

    files_stream = request.files
    if not 'image' in request.files:
        return jsonify({
            "success": False,
            "result": "file not found..."
        })

    f = request.files['image']
    f_name = f.filename

    fpath = os.path.join(IMAGE_TEMP_FOLDER, f_name)
    f.save(fpath)

    yolo.predict(fpath)

    yolo.clear()

    return send_file(fpath)


app.run(port=9999)
