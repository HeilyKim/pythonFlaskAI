import matplotlib
from flask import Flask, request, make_response, send_file, render_template
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import numpy as np
from rf import rf
from flask_cors import CORS
from datetime import datetime
from functools import wraps, update_wrapper

app = Flask(__name__)

CORS(app)


@app.route("/")
def index():
    alcohol = request.args.get("alcohol")
    sugar = request.args.get("sugar")
    pH = request.args.get("pH")
    print(f'alcohol = {alcohol} sugar = {sugar} pH = {pH}')
    if alcohol:
        predValue = rf.predict([[float(alcohol), float(sugar), float(pH)]])
        return str(predValue)
    else:
        return "예측할 값을 입력해라"


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


@app.route('/fig/<int:mean>_<int:var>')
@nocache
def fig(mean, var):
    plt.figure(figsize=(4, 3))
    xs = np.random.normal(mean, var, 100)
    ys = np.random.normal(mean, var, 100)
    plt.scatter(xs, ys, s=100, marker='h', color='red', alpha=0.3)
    # file로 저장하는 것이 아니라 binary object에 저장해서 그대로 file을 넘겨준다고 생각하면 됨
    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)  ## object를 읽었기 때문에 처음으로 돌아가줌
    return send_file(img, mimetype='image/png')
    # plt.savefig(img, format='svg')


@app.route('/fie/<int:a>_<int:b>_<int:c>_<int:d>')
@nocache
def fie(a, b, c, d):
    ratio = [a, b, c, d]
    labels = ['Apple', 'Banana', 'Melon', 'Grapes']
    plt.pie(ratio, labels=labels, autopct='%.1f%%')
    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    return send_file(img, mimetype='image/png')
    # plt.savefig(img, format='svg')


app.run(debug=True, host='0.0.0.0')
