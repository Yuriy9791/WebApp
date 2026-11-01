import os
import argparse
from gds_viewer.ui import make_app
app = make_app()
server = app.server
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8050')))
    args, _ = parser.parse_known_args()
    app.run(host=args.host, port=args.port, debug=False)
