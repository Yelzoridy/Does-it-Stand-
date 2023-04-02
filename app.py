from model import ann
from sklearn.preprocessing import StandardScaler
import pdb
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="templates")
sc = StandardScaler()
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['GET'])
def classify_type():
    try:
        long = request.args.get('long')
        lat = request.args.get('lat')
        floor = request.args.get('floor')
        units = request.args.get('units')
        year = request.args.get('year')
        cement = request.args.get('cement')
        wood = request.args.get('wood')
        metal = request.args.get('metal')
        brick = request.args.get('brick')

        variety = ann.predict(sc.fit_transform([[long, lat, floor, units, year, cement, wood, metal, brick]])) > 0.5
        return render_template('output.html', variety=variety)
    except:
        return 'Error'
