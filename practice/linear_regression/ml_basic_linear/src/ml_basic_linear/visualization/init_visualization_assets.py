import wget
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
url = "https://github.com/GrandmaCan/ML/raw/main/Resgression/ChineseFont.ttf"
output_filename = os.path.join(project_dir, 'resources', 'ChineseFont.ttf')

# Download the file to the specified directory
wget.download(url, out=output_filename)