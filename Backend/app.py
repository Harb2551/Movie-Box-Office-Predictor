from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)


@app.route('/', methods=['GET'])
def g():
    return "hiiiii"


@app.route('/api/query', methods=['POST'])
def get_query_from_react():
    data = request.get_json()
    actor2Name = data['data']['actor_2_name']
    actor1Name = data['data']['actor_1_name']
    actor3Name = data['data']['actor_3_name']
    directorName = data['data']['director_name']
    country = data['data']['country']
    cr = data['data']['cr']
    language = data['data']['language']
    actor1Likes = data['data']['actor_1_likes']
    actor2FacebookLikes = data['data']['actor_2_facebook_likes']
    actor3FacebookLikes = data['data']['actor_3_facebook_likes']
    directorFacebookLikes = data['data']['director_facebook_likes']
    castTotalFacebookLikes = data['data']['cast_total_facebook_likes']
    budget = data['data']['budget']
    gross = data['data']['gross']
    genres = data['data']['genres']
    imdbScore = data['data']['imdb_score']

    print(data)

    result = subprocess.run(['python', 'Model.py', actor1Name, actor2Name, actor3Name, directorName, country, cr, language, actor1Likes, actor2FacebookLikes, actor3FacebookLikes, directorFacebookLikes, castTotalFacebookLikes, budget, gross, genres, imdbScore],
                            )

    f = open('ans.txt', 'r')
    l = list(f.read().split())
    print()
    print(l)
    print()
    d = {}
    k = 1
    for i in l:
        d[k] = i
        k += 1
    dd = {}
    dd[0] = "Predicted Gross Revenue is " + str(l[0])
    dd[1] = "Predicted Movie Rating is " + str(l[1])
    print(dd)
    return dd
