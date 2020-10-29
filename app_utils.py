from flask import abort
import datetime

class app_utils:
  def __init__(self, logger):
    self.logger = logger
    print("Model initialized")

  def check_attributes(self, userdata):
      for x in userdata.keys():
          if x == 'gender' or x == 'statusLevel':  # x == 'city' or x == 'country' or
              if isinstance(userdata[x], str):
                  if len(userdata[x]) == 0:
                      self.logger.error(
                          '{} should not be an empty string User/Poster Attribute'.format(x))
                      abort(
                          400, description='{} should not be an empty string in User/Poster Attribute'.format(x))
              else:
                  self.logger.error('{} should be string'.format(x))
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


  def check_feeditems(self, feeditems):
      for feed in feeditems:
          for key in feed.keys():
              if key == 'feedItemId':
                  if isinstance(feed[key], str):
                      if len(feed[key]) == 0:
                          self.logger.error('Feed id should not be empty')
                          abort(400, description='Feed ID should not be empty')
                  else:
                      self.logger.error("Feed id must be string")
                      abort(400, description='Feed ID must be string')
              elif key == 'numberOfLikes' or \
                      key == 'numberOfComments' or \
                      key == 'postTextLength' or \
                      key == 'numberOfHashTags' or key == 'latitude' or key == 'longitude' or key == 'numberOfMediaUrls':
                  if not isinstance(feed[key], (float, int)):
                      if feed[key] == None:
                          feed[key] = 0.0
                      else:
                          self.logger.error("Value of {} is not appropriate in Feed ID {}".format(
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
                  self.check_attributes(feed[key])
              else:
                  self.logger.error("Please Check payload ")
                  abort(400, description='Please Check payload')


  def check_json(self, data):
      for x in data.keys():
          if x == 'userAttributes':
              self.check_attributes(data['userAttributes'])
          elif x == 'feedItems':
              self.check_feeditems(data['feedItems'])
          else:
              abort(400, description='Data attributes are not correct in the payload')
