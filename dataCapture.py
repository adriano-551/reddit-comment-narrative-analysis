import json
import re
import requests

posts = []

for i in range(30,720,30):
    url = 'https://api.pushshift.io/reddit/search/submission/?subreddit=cardiff&size=500&fields=selftext,is_self&is_video="false"&before=' + str(i) + 'd&after=' + str(i+30) + 'd'
    res = requests.get(url)

    try:
        text = res.json()['data']

        for post in text:
            if (post['selftext'] != '' and post['is_self'] == True and post['selftext'] != '[removed]' and post['selftext'] != '[deleted]' and post['selftext'] != ' '):
                textpost = re.sub(r'http\S+', '', post['selftext'])
                textpost = re.sub("""^A-Za-z0-9 '.,/!?]+""", ' ', textpost)
                textpost = textpost.replace('\n', ' ')
                textpost = textpost.replace('  ', ' ')
            
                posts.append(textpost)
    except:
        continue

onlyText = []

for post in posts:
    onlyText.append({
        'post': post,
        'label': '',
    })

with open("data.json", "w") as output:
    json.dump(onlyText, output)

#("post"){1}(.+?)("isNarrative"){1}