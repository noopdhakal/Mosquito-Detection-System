import sys, os
import shutil
import subprocess
from pathlib import Path

from mosquito.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response, send_from_directory, url_for
from flask_cors import CORS, cross_origin 
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

LIVE_UPLOAD_DIR = Path("data") / "live_uploads"
LIVE_RESULT_DIR = Path("yolov5") / "runs" / "detect" / "live"
LIVE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

clApp = ClientApp()


def get_latest_live_video_url():
    generated_videos = sorted(LIVE_RESULT_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not generated_videos:
        return None
    return url_for("serve_live_result", filename=generated_videos[0].name)


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/live", methods=["GET", "POST"])
@cross_origin()
def liveRoute():
    if request.method == "GET":
        return render_template(
            "live.html",
            status_message="Ready for detection",
            result_video=get_latest_live_video_url(),
        )

    try:
        youtube_url = request.form.get("youtube_url", "").strip()
        video_file = request.files.get("video")
        source = None

        if youtube_url:
            source = youtube_url
        elif video_file is not None and video_file.filename != "":
            filename = secure_filename(video_file.filename)
            input_path = LIVE_UPLOAD_DIR / filename
            video_file.save(str(input_path))
            source = str(input_path.resolve())
        else:
            return render_template(
                "live.html",
                status_message="Please upload a video file or provide a YouTube URL",
                status_type="danger",
            ), 400

        shutil.rmtree(LIVE_RESULT_DIR, ignore_errors=True)
        LIVE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                sys.executable,
                "detect.py",
                "--weights",
                "best.pt",
                "--img",
                "416",
                "--conf",
                "0.5",
                "--source",
                source,
                "--project",
                "runs/detect",
                "--name",
                "live",
                "--exist-ok",
            ],
            cwd="yolov5",
            check=True,
        )

        latest_video_url = get_latest_live_video_url()
        if not latest_video_url:
            return render_template(
                "live.html",
                status_message="Detection completed but no output video was generated",
                status_type="warning",
            ), 500

        return render_template(
            "live.html",
            result_video=latest_video_url,
            status_message="Detection completed successfully",
            status_type="success",
        )

    except subprocess.CalledProcessError as exc:
        print(exc)
        return render_template(
            "live.html",
            status_message="Video detection failed. Check server logs for details.",
            status_type="danger",
        ), 500
    except Exception as exc:
        print(exc)
        return render_template(
            "live.html",
            status_message="Invalid video input",
            status_type="danger",
        ), 400


@app.route("/live/results/<path:filename>")
def serve_live_result(filename):
    return send_from_directory(LIVE_RESULT_DIR, filename)


@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")

        opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"
    finally:
        shutil.rmtree("yolov5/runs", ignore_errors=True)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)