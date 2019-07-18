import base64
import io
import json
import os

import pandas as pd
from PIL import Image, ImageDraw
import PIL
from flask import Flask, request

from .image_processing import get_image_tensor, pil_to_cv, get_approx
from .model_utils import get_model_from_file, predict_segmented_image, BuildingSegmenterNet
from .data_processing import unsqueeze_approx, offset_approx, generate_osm_xml

app = Flask(__name__)
app.config.from_object(__name__)
app.config.update({"data_dir":"autoboundData", "data_csv": "data.csv", "original_image_dir": "originalImages",
                   "segmented_image_dir": "segmentedImages", "id":-1})


def file_exists(filename):
    return os.path.isfile(filename)

def save_to_file(text):
    f = open('dummy.txt', 'w+')
    f.write(text)
    f.close()

@app.route('/', methods=['GET', 'POST'])
def foo():
    data = json.loads(request.data)
    image_bytes = base64.b64decode(data['image'])
    image = get_image_tensor(io.BytesIO(image_bytes))
    model = get_model_from_file(filename=os.path.join(app.root_path, 'models', 'autoboundModel.pth'))
    segmented_image = predict_segmented_image(model, image)
    segmented_image = pil_to_cv(segmented_image)
    approx = get_approx(segmented_image)
    approx = unsqueeze_approx(approx)
    approx = offset_approx(approx, float(data['min_east']), float(data['min_north']))
    osm_xml, id = generate_osm_xml(approx, app.config['id'])
    app.config.update({"id":id})
    return osm_xml


@app.route('/generateNodesTest', methods=['GET', 'POST'])
def generate_nodes_test():
    data = json.loads(request.data)
    if data['verification'] == "generateNodesTest":
        return "message exchange works"
    return "message error problem"


@app.route('/dataCollector', methods=['GET', 'POST'])
def data_collector():
    data = json.loads(request.data)
    data_dir = app.config['data_dir']
    original_image_dir = os.path.join(data_dir, app.config['original_image_dir'])
    segmented_image_dir = os.path.join(data_dir, app.config['segmented_image_dir'])
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if not file_exists(os.path.join(data_dir,app.config['data_csv'])):
        df = pd.DataFrame(columns=["timestamp", "id", "image_suffix", "min_east", "min_north",
                                   "max_east", "max_north", "dist100pixel"])
    else:
        df = pd.read_csv(os.path.join(data_dir, app.config['data_csv']))
    img_filename = str(data['timestamp'])
    while os.path.exists(os.path.join(original_image_dir, img_filename + '.png')):
        img_filename = str(int(img_filename)+1)
    entry = []
    way = data['way']
    entry.extend([int(data['timestamp']), int(way['id']), img_filename, float(data['minEast']), float(data['minNorth']),
                  float(data['maxEast']), float(data['maxNorth']), float(data['dist100pixel'])])
    df.loc[len(df) + 1] = entry
    df.to_csv(os.path.join(data_dir, app.config['data_csv']), index = False)
    image_bytes = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_bytes))
    if not os.path.isdir(original_image_dir):
        os.makedirs(original_image_dir)
    image.save(os.path.join(original_image_dir, img_filename+'.png'), 'png')
    segmented_image = Image.new('RGB', image.size, 'black')
    # Use ImageDraw.polygon
    nodes = data['way']['nodes']
    # calculate ratios reating pixels to east north
    east_ratio = image.size[0]/(float(data['maxEast'])-float(data['minEast']))
    north_ratio = image.size[1] / (float(data['maxNorth']) - float(data['minNorth']))
    vertices = []
    for node in nodes:
        node_east = east_ratio*(float(node['east'])-float(data['minEast']))
        node_north = north_ratio*(float(node['north'])-float(data['minNorth']))
        vertices.append((int(node_east), int(node_north)))
    draw = ImageDraw.Draw(segmented_image)
    draw.polygon(vertices, (255, 255, 255), (255, 255, 255))
    segmented_image = segmented_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    if not os.path.isdir(segmented_image_dir):
        os.makedirs(segmented_image_dir)
    segmented_image.save(os.path.join(segmented_image_dir, img_filename+'_segmented.png'), 'png')
    return 'success'
