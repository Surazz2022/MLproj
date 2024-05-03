import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)