from flask import Flask, request, session, g, redirect, url_for, abort, \
render_template, flash, make_response

app = Flask(__name__) 
app.config.from_object(__name__)

@app.route('/', methods=['GET', 'POST'])
def foo():
    return "Lorem Ipsum"