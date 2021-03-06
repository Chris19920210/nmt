import tcelery
import align_tasks_enzh
import argparse
import json
import logging
import traceback
from tornado import web, ioloop, gen
from logging.handlers import TimedRotatingFileHandler
import os
"""Tornado Web Application"""

parser = argparse.ArgumentParser(description='nmt web application')
parser.add_argument('--host', type=str, default=None,
                    help='host')
parser.add_argument('--port', type=int, default=None,
                    help='port')
args = parser.parse_args()

tcelery.setup_nonblocking_producer()


class MyAppException(web.HTTPError):

    pass


class MyAppBaseHandler(web.RequestHandler):

    def write_error(self, status_code, **kwargs):

        self.set_header('Content-Type', 'application/json')
        if self.settings.get("serve_traceback") and "exc_info" in kwargs:
            # in debug mode, try to send a traceback
            lines = []
            for line in traceback.format_exception(*kwargs["exc_info"]):
                lines.append(line)
            self.finish(json.dumps({
                'error': {
                    'code': status_code,
                    'message': self._reason,
                    'traceback': lines,
                }
            }))
        else:
            self.finish(json.dumps({
                'error': {
                    'code': status_code,
                    'message': self._reason,
                }
            }))


class AsyncAppNmtHandler(MyAppBaseHandler):
    global logger

    @web.asynchronous
    @gen.coroutine
    def get(self):
        content_type = self.request.headers.get('Content-Type')
        if not (content_type and content_type.lower().startswith('application/json')):
            MyAppException(reason="Wrong data format, needs json", status_code=400)
        logger.info(self.request.body.decode("utf-8"))
        res = yield gen.Task(align_tasks_enzh.alignment.apply_async, args=[self.request.body])
        ret = res.result
        self.write(ret)
        logger.info(ret)
        self.finish()

    @web.asynchronous
    @gen.coroutine
    def post(self):
        content_type = self.request.headers.get('Content-Type')
        if not (content_type and content_type.lower().startswith('application/json')):
            MyAppException(reason="Wrong data format, needs json", status_code=400)
        logger.info(self.request.body.decode("utf-8"))
        res = yield gen.Task(align_tasks_enzh.alignment.apply_async, args=[self.request.body])
        ret = res.result
        self.write(ret)
        logger.info(ret)
        self.finish()


if __name__ == '__main__':
    log_format = "%(asctime)s-%(levelname)s-%(message)s"
    os.makedirs("align_enzh_logs", exist_ok=True)
    handler = TimedRotatingFileHandler('align_enzh_logs/alignment_enzh.log', when='midnight', interval=1,
                                       encoding="utf-8")
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    handler.suffix = "%Y%m%d"

    logger = logging.getLogger("alignment_enzh")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    application = web.Application([(r"/alignment", AsyncAppNmtHandler)])
    application.listen(port=args.port, address=args.host)
    ioloop.IOLoop.instance().start()
