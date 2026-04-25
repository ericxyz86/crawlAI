from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    print('Starting minimal Flask app on http://127.0.0.1:5002')
    app.run(host='127.0.0.1', port=5002, debug=True)