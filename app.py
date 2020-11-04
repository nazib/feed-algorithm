import os
import flask
from flask import jsonify
from model.NonLinearModel import NonLinearModel
from gevent.pywsgi import WSGIServer
from app_utils import app_utils, use_gcloud_logging

def create_app(config_filename):
    app = flask.Flask(__name__)
    # app.config.from_pyfile(config_filename)
    utils = app_utils(app.logger)
    nonlin_model = NonLinearModel()
    if not nonlin_model.istrained:
        raise Exception("no trained model found")
    app.logger.info("Model initiated " + nonlin_model.model_path)

    @app.errorhandler(400)
    def value_error(e):
        return jsonify(error=str(e)), 400

    @app.route('/health', methods=['GET'])
    def health_check():
        status = {200: "Container running successfully"}
        return jsonify(status)

    @app.route('/nonlinear/global_rank', methods=['POST'])
    def NonGRank():
        data = flask.request.get_json(force=True)
        utils.check_json(data)
        _, ranks = nonlin_model.GlobalRank(data["feedItems"])
        json_obj = {}
        json_obj["feedItemsRank"] = [x for x in ranks.values()]
        return jsonify(json_obj)

    @app.route('/nonlinear/personal_rank', methods=['POST'])
    def NonPRank():
        data = flask.request.get_json(force=True)
        utils.check_json(data)
        ranks = nonlin_model.PersonalRank(data)
        json_obj = {}
        json_obj["feedItemsRank"] = [x for x in ranks.values()]
        return jsonify(json_obj)

    return app


if __name__ == "__main__":
    #use_gcloud_logging()
    port = int(os.getenv('PORT', 8080))
    http_server = WSGIServer(('0.0.0.0', port), create_app('production'))
    http_server.serve_forever()
