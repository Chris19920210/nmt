from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import simplejson as json
import logging

from .serving_utils import EnZhAlignClient


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("server", None, "Address to Tensorflow Serving server.")
flags.DEFINE_string("servable_name", None, "Name of served model.")
flags.DEFINE_string("problem", None, "Problem name.")
flags.DEFINE_string("data_dir", None, "Data directory, for vocab files.")
flags.DEFINE_string("t2t_usr_dir", None, "Usr dir for registrations.")
flags.DEFINE_string("user_dict", None, "user defined dict path")
flags.DEFINE_integer("timeout_secs", 100, "Timeout for query.")
flags.DEFINE_string("port", None, "Port")
flags.DEFINE_string("host", None, "host")

app = Flask(__name__)
CORS(app)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def validate_flags():
    """Validates flags are set to acceptable values."""
    assert FLAGS.server
    assert FLAGS.servable_name
    assert FLAGS.port


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/alignment", methods=['POST', 'GET'])
def alignment():
    global align_client
    try:
        data = json.loads(request.get_data(), strict=False)
        return json.dumps(align_client.query(data), indent=1, ensure_ascii=False)
    except Exception as e:
        logging.error(str(e))
        raise InvalidUsage('Ooops. Something went wrong', status_code=503)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename='./gnmt_query.log',
                        filemode='w')
    align_client = EnZhAlignClient(
        FLAGS.t2t_usr_dir,
        FLAGS.problem,
        FLAGS.data_dir,
        FLAGS.user_dict,
        FLAGS.server,
        FLAGS.servable_name,
        FLAGS.timeout_secs
    )

    print("Starting app...")
    app.run(host=FLAGS.host, threaded=True, port=FLAGS.port)
