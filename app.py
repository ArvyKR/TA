from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
from flask_mysqldb import MySQL
import yaml
from datetime import datetime
import pytz


app = Flask(__name__)
CORS(app)

db = yaml.full_load(open("db.yaml"))
app.config["MYSQL_HOST"] = db["mysql_host"]
app.config["MYSQL_USER"] = db["mysql_user"]
app.config["MYSQL_PASSWORD"] = db["mysql_password"]
app.config["MYSQL_DB"] = db["mysql_db"]

mysql = MySQL(app)

desired_timezone = pytz.timezone("Asia/Jakarta")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    timestamp = datetime.now(desired_timezone)
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO chat(input, response, timestamp) VALUES (%s, %s, %s)",
        (text, response, timestamp),
    )
    mysql.connection.commit()
    cur.close()
    return jsonify(message)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/things-to-do")
def thingstodo():
    return render_template("thingstodo.html")


@app.route("/gallery")
def gallery():
    return render_template("gallery.html")


@app.route("/history")
def history():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM chat")
    data = cur.fetchall()
    return render_template("history.html", chat=data)


if __name__ == "__main__":
    app.run(debug=True)
