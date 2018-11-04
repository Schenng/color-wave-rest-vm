from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World 2!"

@app.route('/image')
def image():
    return "Image Response"

if __name__ == '__main__':
    #app.run(debug=True)
    app.run()
