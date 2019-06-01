import json

from flask import Flask, request

app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/', methods=['GET', 'POST'])
def foo():
    return "Lorem Ipsum"


@app.route('/generateNodesTest', methods=['GET', 'POST'])
def generate_nodes_test():
    data = request.args.get('data')
    data = json.loads(data)
    if(data['verification']=="generateNodesTest"):
        return "message exchange works"
    return "message error problem"
