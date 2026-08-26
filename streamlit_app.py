import sys
import os

# Set root directory in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the UI app
from ui.streamlit_app import *
