from flask import Flask, request

app = Flask(__name__)

@app.route('/request')
def req():
    word = request.args.get("word")
    

if __name__ == '__main__':
    app.run()
