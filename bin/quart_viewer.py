import asyncio
import base64
import cv2
from dash import Dash, dcc, html, ctx
import threading

from dash.dependencies import Output, Input
from quart import Quart, websocket
from dash_extensions import WebSocket

from pathlib import Path

from turbx import REPO_PATH
from turbx.vis import VideoStream

server = Quart(__name__)
frame_rate = 20
frame_delay = 1.0 / frame_rate
frame_idx = 0


async def video_stream(video, ws_id, is_command=False):
    @server.websocket(f"{ws_id}")
    async def stream():
        global frame_idx
        while True:
            if frame_delay is not None:
                await asyncio.sleep(frame_delay)
            frame = video[frame_idx]
            if is_command:
                frame_idx += 1
            await websocket.send(
                f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}"
            )

    await stream()


@server.websocket("/original")
async def original_stream():
    camera = VideoStream(f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4")
    while True:
        if frame_delay is not None:
            await asyncio.sleep(frame_delay)  # add delay if CPU usage is too high
        frame = camera.get_frame()
        await websocket.send(
            f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}"
        )


@server.websocket("/label")
async def label_stream():
    camera = VideoStream(f"{REPO_PATH}/data/mp4/2010-09-08_074500_HF_S001.mp4")
    while True:
        if frame_delay is not None:
            await asyncio.sleep(frame_delay)  # add delay if CPU usage is too high
        frame = camera.get_frame()
        await websocket.send(
            f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}"
        )


# Create small Dash application for UI.
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__)

app.layout = html.Div(
    style={"height": "100%", "width": "100%"},
    children=[
        html.Div(children=["Turbx Visualization"], style={"textAlign": "center"}),
        dcc.Input(
            id="data_path",
            type="text",
            placeholder="Input path to mp4s",
            style={"width": "100%"},
        ),
        html.Div(
            children=[html.Div("Choose video file: "), dcc.Dropdown(id="files-list"),]
        ),
        html.Div(
            children=[
                html.Button(
                    "Generate filter",
                    id="filter-button",
                    n_clicks=0,
                    style={"background-color": "white"},
                )
            ]
        ),
        html.Div(
            className="row",
            style={"height": "70vh"},
            children=[
                html.Div(
                    className="column",
                    style={
                        "width": "50%",
                        "padding": 10,
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "textAlign": "center",
                    },
                    id="Original Video",
                    children=[
                        html.Img(id="original_video", style={"height": "70vh"},),
                        WebSocket(
                            url=f"ws://127.0.0.1:5000/original", id="ws-original"
                        ),
                    ],
                ),
                html.Div(
                    className="column",
                    style={
                        "width": "50%",
                        "padding": 10,
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "textAlign": "center",
                    },
                    id="Label Video",
                    children=[
                        html.Img(id="label_video", style={"height": "70vh"},),
                        WebSocket(url=f"ws://127.0.0.1:5000/label", id="ws-label"),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("files-list", "options"),
    Output("files-list", "value"),
    Input("data_path", "value"),
)
def cb_update_file_list(folder):
    files_list = [str(x.name) for x in Path(f"{folder}").glob("*.mp4")]
    if len(files_list) > 0:
        value = files_list[0]
    else:
        value = "No files found"
    return files_list, value


@app.callback(Output("filter-button", "style"), Input("filter-button", "n_clicks"))
def start_filter_on_click(filter_button):
    if filter_button % 2 == 0:
        return {"background-color": "white"}
    else:
        return {"background-color": "green"}


# Copy data from websocket to Img element.
app.clientside_callback(
    """
    function(m) {
        return m? m.data : '';
    }
    """,
    Output(f"original_video", "src"),
    Input(f"ws-original", "message"),
)


app.clientside_callback(
    """
    function(m) {
        return m? m.data: '';
    }
    """,
    Output(f"label_video", "src"),
    Input(f"ws-label", "message"),
)

if __name__ == "__main__":
    threading.Thread(target=app.run_server).start()
    server.run()
