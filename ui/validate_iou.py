import json
from datapreparation.utils.iou_validator import find_low_iou_pairs
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    with open('metadata.jsonl') as f:
        entries = [json.loads(line) for line in f]
    return render_template('index.html', entries=entries)

@app.route('/review/<img_id>', methods=['GET', 'POST'])
def review(img_id):
    gt = json.load(open(f'annotations/{img_id}.json'))
    auto = json.load(open(f'output/auto_labels/{img_id}.json'))
    low_iou = find_low_iou_pairs(gt, auto)
    if request.method == 'POST':
        corrections = request.json
        with open(f'annotations/{img_id}.json', 'w') as f:
            json.dump(corrections, f)
        return '', 204
    return render_template('review.html', img_id=img_id, low_iou=low_iou)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
