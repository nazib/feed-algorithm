import flask
from flask import jsonify, abort
from .NonLinearModel import NonLinearModel
from gevent.pywsgi import WSGIServer
import datetime
import os

app = flask.Flask(__name__)
nonlin_model = None


def check_attributes(userdata):
    for x in userdata.keys():
        if x == 'gender' or x == 'statusLevel':  # x == 'city' or x == 'country' or
            if isinstance(userdata[x], str):
                if len(userdata[x]) == 0:
                    app.logger.error(
                        '{} should not be an empty string User/Poster Attribute'.format(x))
                    abort(
                        400, description='{} should not be an empty string in User/Poster Attribute'.format(x))
            else:
                app.logger.error('{} should be string'.format(x))
                abort(
                    400, description='{} should be string in User/ Poster Attribute'.format(x))

        if x == "totalReceivedPostComments" or x == "totalReceivedPostLikes" or x == "numberOfFollowers":
            if not isinstance(userdata[x], (float, int)):
                if userdata[x] == None:
                    userdata[x] = 0.0
                else:
                    app.logger.error("{} should be float value".format(x))
                    abort(
                        400, description='{} should be float value in User/Poster Attribute'.format(x))


def check_feeditems(feeditems):
    for feed in feeditems:
        for key in feed.keys():
            if key == 'feedItemId':
                if isinstance(feed[key], str):
                    if len(feed[key]) == 0:
                        app.logger.error('Feed id should not be empty')
                        abort(400, description='Feed ID should not be empty')
                else:
                    app.logger.error("Feed id must be string")
                    abort(400, description='Feed ID must be string')
            elif key == 'numberOfLikes' or \
                    key == 'numberOfComments' or \
                    key == 'postTextLength' or \
                    key == 'numberOfHashTags' or key == 'latitude' or key == 'longitude' or key == 'numberOfMediaUrls':
                if not isinstance(feed[key], (float, int)):
                    if feed[key] == None:
                        feed[key] = 0.0
                    else:
                        app.logger.error("Value of {} is not appropriate in Feed ID {}".format(
                            key, feed['feedItemId']))
                        abort(400, description="Value of {} is not appropriate in Feed ID {}".format(
                            key, feed['feedItemId']))
            elif key == "postedDate":
                if not isinstance(feed[key], str):
                    abort(400, description="Value of {} must be string in Feed ID {}".format(
                        key, feed['feedItemId']))
                else:
                    try:
                        datetime.datetime.strptime(
                            feed[key], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        abort(400, description="Date time format is not correct in Feed ID {}".format(
                            feed['feedItemId']))
            elif key == 'posterAttributes':
                check_attributes(feed[key])
            else:
                app.logger.error("Please Check payload ")
                abort(400, description='Please Check payload')


def check_json(data):
    for x in data.keys():
        if x == 'userAttributes':
            check_attributes(data['userAttributes'])
        elif x == 'feedItems':
            check_feeditems(data['feedItems'])
        else:
            abort(400, description='Data attributes are not correct in the payload')


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
    _, ranks = nonlin_model.GlobalRank(data["feedItems"])
    json_obj = {}
    json_obj["feedItemsRank"] = [x for x in ranks.values()]
    return jsonify(json_obj)


@app.route('/nonlinear/personal_rank', methods=['POST'])
def NonPRank():
    data = flask.request.get_json(force=True)
    check_json(data)
    ranks = nonlin_model.PersonalRank(data)
    json_obj = {}
    json_obj["feedItemsRank"] = [x for x in ranks.values()]
    return jsonify(json_obj)


if __name__ == "__main__":
    nonlin_model = NonLinearModel()
    port = int(os.getenv('PORT', 8080))
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
