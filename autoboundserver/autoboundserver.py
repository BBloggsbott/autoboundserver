import json
import os

import pandas as pd
from flask import Flask, request

app = Flask(__name__)
app.config.from_object(__name__)
app.config.update({"data_dir":"autoboundData", "data_csv": "data.csv", "original_image_dir": "originalImages",
                   "segmented_image_dir": "segmentedImages"})


def file_exists(filename):
    return os.path.isfile(filename)


@app.route('/', methods=['GET', 'POST'])
def foo():
    return "Lorem Ipsum"


@app.route('/generateNodesTest', methods=['GET', 'POST'])
def generate_nodes_test():
    data = request.args.get('data')
    data = json.loads(data)
    if data['verification'] == "generateNodesTest":
        return "message exchange works"
    return "message error problem"


# TODO : Save image and segmented image
@app.route('/dataCollector', methods=['GET', 'POST'])
def data_collector():
    data = request.args.get("data")
    data = json.loads(data)
    if not file_exists(os.path.join(app.config['data_dir'],app.config['data_csv'])):
        df = pd.DataFrame(columns=["timestamp", "id", "image_suffix", "min_east", "min_north",
                                   "max_east", "max_north", "dist100pixel"])
    else:
        df = pd.read_csv(os.path.join(app.config['data_dir'], app.config['data_csv']))
    img_filename=""
    entry = []
    way = data['way']
    entry.extend([int(data['timestamp']), way['id'], img_filename, data['minEast'], data['minNorth'],
                  data['maxEast'], data['maxNorth'], data['dist100pixel']])
    row = df.shape[0]
    df.loc[row] = entry
    df.to_csv(os.path.join(app.config['data_dir'], app.config['data_csv']), index=False)
