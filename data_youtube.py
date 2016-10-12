#!/usr/bin/python

from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser
import datetime
import pickle

# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = "AIzaSyDrSl4I4UHm9aqY2DzAM6hGQ82z1wPobT4"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def youtube_search(options):
  # videos_dict = pickle.load( open( "videos_dict.p", "rb" ) )
  youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

  # Call the search.list method to retrieve rseults matching the specified
  # query term.
  search_response = youtube.search().list(
    q=options.q,
    type="video",
    location=options.location,
    locationRadius=options.location_radius,
    part="id,snippet",
    maxResults=options.max_results
  ).execute()

  search_videos = []

  # Merge video ids
  for search_result in search_response.get("items", []):
    search_videos.append(search_result["id"]["videoId"])
  video_ids = ",".join(search_videos)

  # Call the videos.list method to retrieve location details for each video.
  video_response = youtube.videos().list(
    id=video_ids,
    part='snippet, recordingDetails'
  ).execute()

  videos = []

  # Add each result to the list, and then display the list of matching videos.
  snum = 0
  videos_dict = {}
  for video_result in video_response.get("items", []):    
    # videos.append("%s, (%s,%s)" % (video_result["snippet"]["title"],
    #                           video_result["recordingDetails"]["location"]["latitude"],
    #                           video_result["recordingDetails"]["location"]["longitude"]))

    title = str(video_result["snippet"]["title"])
    description = video_result["snippet"]["localized"]["description"].encode('utf-8')
    upload_date_str = video_result["snippet"]["publishedAt"].encode('utf-8')
    upload_date_str = upload_date_str[0:10]
    upload_date = datetime.datetime.strptime(upload_date_str, '%Y-%m-%d').date()
    latitude = float(str(video_result["recordingDetails"]["location"]["latitude"]))
    longitude = float(str(video_result["recordingDetails"]["location"]["longitude"]))
    
    videos_dict[snum] = [title, description, upload_date, latitude, longitude]
    snum += 1


  print "Dict : "
  print videos_dict
  pickle.dump( videos_dict, open( "videos_dict.p", "wb" ) )
  # print "Videos:\n", "\n".join(videos), "\n"


if __name__ == "__main__":
  argparser.add_argument("--q", help="Search term", default="2011 Mississippi River Floods")
  argparser.add_argument("--location", help="Location", default="25.953807, -97.576315")
  argparser.add_argument("--location-radius", help="Location radius", default="999km")
  argparser.add_argument("--max-results", help="Max results", default=50)
  args = argparser.parse_args()

  try:
    youtube_search(args)
  except HttpError, e:
    print "An HTTP error %d occurred:\n%s" % (e.resp.status, e.content)

