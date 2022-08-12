import asyncio
import base64
import dash, cv2
import dash_html_components as html
import threading

from dash.dependencies import Output, Input
from quart import Quart, websocket
from dash_extensions import WebSocket

from turbx import REPO_PATH

server = Quart(__name__)
frame_rate = 20
frame_delay = 1.0 / frame_rate

# WIP - NOT WORKING
# TODO: implement simple version - static everything


class VideoStream(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.max_idx = self.video.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        self.idx = self.video.get(cv2.CAP_PROP_POS_FRAMES)

    def __del__(self):
        self.video.release()

    def get_frame(self):

        # loop video
        self.idx = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        if self.idx == self.max_idx:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # return byte-encoded image
        success, image = self.video.read()
        if success:
            cv2.putText(
                img=image,
                text=f"{int(self.idx)}/{int(self.max_idx)}",
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=1,
            )
            _, jpeg = cv2.imencode(".jpg", image)
            return jpeg.tobytes()
        else:
            raise IOError("Failed to retrieve video frame.")


@server.websocket("/original_stream")
async def original_stream():
    camera = VideoStream(
        f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4"
    )  # zero means webcam
    while True:
        if frame_delay is not None:
            await asyncio.sleep(frame_delay)  # add delay if CPU usage is too high
        frame = camera.get_frame()
        await websocket.send(
            f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}"
        )


@server.websocket("/label_stream")
async def label_stream():
    camera = VideoStream(
        f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S002_S001.mp4"
    )  # zero means webcam
    while True:
        if frame_delay is not None:
            await asyncio.sleep(frame_delay)  # add delay if CPU usage is too high
        frame = camera.get_frame()
        await websocket.send(
            f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}"
        )


# Create small Dash application for UI.
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Img(style={"width": "20%", "padding": 10}, id="original_video"),
        html.Img(style={"width": "20%", "padding": 10}, id="label_video"),
        WebSocket(url=f"ws://127.0.0.1:5000/original_stream", id="ws-original"),
        WebSocket(url=f"ws://127.0.0.1:5001/label_stream", id="ws-label"),
    ],
)

# Copy data from websocket to Img element.
app.clientside_callback(
    "function(m){return m? m.data : '';}",
    [Output(f"original_video", "src"), Output(f"label_video", "src")],
    [Input(f"ws-original", "message"), Input(f"ws-label", "message")],
)

if __name__ == "__main__":
    threading.Thread(target=app.run_server).start()
    server.run()
