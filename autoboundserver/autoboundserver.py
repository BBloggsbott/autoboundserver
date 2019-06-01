from flask import Flask

app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/', methods=['GET', 'POST'])
def foo():
    return "Lorem Ipsum"
