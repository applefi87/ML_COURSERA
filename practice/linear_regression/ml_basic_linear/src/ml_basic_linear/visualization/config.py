import os

# PROJECT_PATH= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_project_root():
    """Returns the path to the project root folder."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
