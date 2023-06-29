from flask import Flask, render_template, request
import KNN

app = Flask(__name__)


@app.route("/")
def halo():
    return "halo"


@app.route("/aa", methods=["GET", "POST"])
def aa():
    result = 10
    if request.method == "POST":
        x = request.form['x']
        y = request.form['y']
        result = KNN.doPred(int(x),int(y))
    return render_template("aa.html",result=result)


app.run(host='127.0.0.1', debug=True)
