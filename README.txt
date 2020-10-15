This program uses the youtube api to collect video data from Anthony Fantano's youtube channel "Theneedledrop".
It uses youtube analytics and the video's description to predict the score that he will give to an album.
Within the video_scrape file, I use an environmental variable to locate the value for my api key. If you insert your own key in it's place, the python files will be able to run on their own and create the model.

Title: The title of the video
Description: The information in the youtube description
like_count: count of likes
dislike_count: count of dislikes
view_count: view count in thousands
duration: video length in seconds
comment_count: count of comments on video
