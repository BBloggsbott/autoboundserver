from autoboundserver import app, BuildingSegmenterNet
import os

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
app.run(host='127.0.0.1', port=port, debug=True)
