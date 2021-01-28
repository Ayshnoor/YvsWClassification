from flask import Flask, render_template, request
from Y_W_model_class import y_w_model

app = Flask(__name__)


@app.route('/', methods =['GET', 'POST'])
def home():
    return render_template('home.html')
@app.route('/answer', methods =['GET', 'POST'])
def answer():
    render_template('answer.html')
    if request.args.get('reddit_submission'):
        reddit_submission = request.args.get('reddit_submission')
        y_pred = y_w_model(reddit_submission)

        if y_pred == 1:
            return render_template('yoga.html')
        else:
            return render_template('weightlifting.html')
    else:
        return render_template('answer.html')

if __name__ == '__main__':
    app.run(debug=True)
