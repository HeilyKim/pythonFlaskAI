from flask import Flask, request
from rf import rf

app = Flask(__name__)


@app.route("/")
def index():
    alcohol = request.args.get("alcohol")
    sugar = request.args.get("sugar")
    pH = request.args.get("pH")
    if alcohol:
        predValue = rf.predict([[float(alcohol),float(sugar),float(pH)]])
        return str(predValue)
    else:
        return "예측할 값을 입력해라"


app.run(debug=True)
