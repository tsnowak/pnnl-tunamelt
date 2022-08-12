from flask import Flask, render_template, Response
from turbx import REPO_PATH

# WIP - WORKING


class VideoStream(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        _, jpeg = cv2.imencode(".jpg", image)
        return jpeg.tobytes()


app = Flask(
    __name__,
    static_folder=f"{REPO_PATH}/data",
    template_folder=f"{REPO_PATH}/templates",
)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
