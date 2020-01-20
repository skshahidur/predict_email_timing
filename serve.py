from flask import Flask, json
from pairfinance import predictor
import flask

api = Flask(__name__)

@api.route('/pairfinance', methods=['GET','POST'])
def post_companies():
    json_file = flask.request.json
    return predictor(json_file)

if __name__ == '__main__':
    api.run()

