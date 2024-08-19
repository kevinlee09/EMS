from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    ROOT_PATH = "/home/eyebrow-matting/"
elif platform == "win32":
    # Windows...
    ROOT_PATH = "G:/eyebrow-matting/"


