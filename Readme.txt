1. heroku login
2. heroku git:remote -a covid-19-classification

$ git add .
$ git commit -am "make it better"
$ git push heroku master

export FLASK_APP=flask_server.py
flask run