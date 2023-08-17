# !pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="b0YiGjFTyMDpGi8mnhEx")
project = rf.workspace("roboflow-gw7yv").project("fish-yzfml")
dataset = project.version(44).download("yolov8")
