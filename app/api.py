import os
import uuid
import puppercnn
print("CARGO PUPPERCNN")
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from response import Response

TEMP_PATH = "temp/"

if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

def extract_file_ext(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension

class Upload(Resource):
    allowed_file_exts = ('.jpg','.png', '.jpge')
    fail_response = Response()

    @staticmethod
    def allowed_file(filename):
        return extract_file_ext(filename) in Upload.allowed_file_exts

    def post(self):
        try:
            file = request.files['image']
            if file and self.allowed_file(file.filename):
                temp_filename = str(uuid.uuid4()) + extract_file_ext(file.filename)
                temp_filepath = os.path.join(TEMP_PATH, temp_filename)
                print('saving:', temp_filepath)
                file.save(temp_filepath)
                response = puppercnn.predict_breed(temp_filepath)
                print(response)
                return jsonify(vars(response))
                # return jsonify({
                #     "result": True
                # })
            else:
                raise Exception('File extension not allowed')
        except Exception as e:
            print(e)
            return jsonify(vars(Upload.fail_response))
        finally:
            try:
                print('removing:', temp_filepath)
                os.remove(temp_filepath)
            except Exception as e:
                print(e)

if __name__ == '__main__':
    host = 'localhost'
    port = 3000
    app = Flask(__name__)

    api = Api(app)
    api.add_resource(Upload, '/api/upload')
    app.run(host=host, port=port)