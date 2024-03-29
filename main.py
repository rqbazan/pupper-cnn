import os
import uuid
import logging as log
import puppercnn
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from response import Response

log.basicConfig(filename='pupper-cnn.log', level=log.ERROR)

TEMP_PATH = "temp/"

if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)


def extract_file_ext(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension


class Upload(Resource):
    allowed_file_exts = ('.jpg', '.png', '.jpeg')
    fail_response = Response()

    @staticmethod
    def allowed_file(filename):
        return extract_file_ext(filename) in Upload.allowed_file_exts

    def post(self):
        try:
            temp_filepath = None
            file = request.files['image']
            if file and self.allowed_file(file.filename):
                temp_filename = str(uuid.uuid4()) + extract_file_ext(file.filename)
                temp_filepath = os.path.join(TEMP_PATH, temp_filename)
                print('saving: {}'.format(temp_filepath))
                log.debug('saving: {}'.format(temp_filepath))
                file.save(temp_filepath)
                response = puppercnn.predict_breed(temp_filepath)
                return jsonify(vars(response))
            else:
                raise Exception('File extension not allowed')
        except Exception as e:
            log.error(e)
            return jsonify(vars(Upload.fail_response))
        finally:
            try:
                if temp_filepath:
                    print('removing: {}'.format(temp_filepath))
                    log.debug('removing: {}'.format(temp_filepath))
                    os.remove(temp_filepath)
            except Exception as e:
                log.error(e)

host = '0.0.0.0'
port = 8000
app = Flask(__name__)

api = Api(app)
api.add_resource(Upload, '/api/upload')
app.run(port=port, host=host)
