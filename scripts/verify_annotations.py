import webbrowser

def review_flagged(flagged_path='data/flagged_cases.txt'):
    with open(flagged_path) as f:
        for line in f:
            _, image_path, _ = line.strip().split(',')
            webbrowser.open(image_path)
