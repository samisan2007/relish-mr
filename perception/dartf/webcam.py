"""Stream the Windows webcam into the RTX 3080 FAST container and preview tracks."""

import argparse
from pathlib import Path
import subprocess
import threading
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', nargs='?', default='pen')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--frames', type=int, default=300)
    args = parser.parse_args()
    if not args.prompt.strip() or ',' in args.prompt or args.frames < 1:
        parser.error('Use one non-empty prompt and a positive frame count')

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}; close other camera apps or try --camera 1')

    script = Path(__file__).with_name('rtx3080.ps1')
    process = subprocess.Popen(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', str(script), '-Action', 'Live', '-Prompt', args.prompt, '-Frames', str(args.frames)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    stop = threading.Event()
    latest = [None]
    tracks = [()]

    def capture():
        while not stop.is_set():
            ok, frame = camera.read()
            if not ok:
                stop.set()
                break
            latest[0] = frame

    def send():
        try:
            while not stop.is_set() and process.poll() is None:
                frame = latest[0]
                if frame is None:
                    time.sleep(.01)
                    continue
                ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    process.stdin.write(jpg.tobytes())
                time.sleep(1 / 30)
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()

    def output():
        frame_id, boxes = 0, []
        for raw in process.stdout:
            line = raw.decode(errors='replace').strip()
            fields = line.split(',')
            if len(fields) == 10 and fields[0].isdigit():
                try:
                    number, track_id = int(fields[0]), int(fields[1])
                    box = tuple(map(float, fields[2:6]))
                except ValueError:
                    continue
                if number != frame_id:
                    frame_id, boxes = number, []
                boxes.append((track_id, box))
                tracks[0] = (time.monotonic(), tuple(boxes))
            elif line:
                print(line, flush=True)

    workers = [threading.Thread(target=target, daemon=True) for target in (capture, send, output)]
    for worker in workers:
        worker.start()
    print('FAST webcam starting. Press Q in the preview to stop.', flush=True)
    try:
        while process.poll() is None and not stop.is_set():
            frame = latest[0]
            if frame is not None:
                preview = frame.copy()
                if tracks[0] and time.monotonic() - tracks[0][0] < 1:
                    for track_id, (x, y, w, h) in tracks[0][1]:
                        cv2.rectangle(preview, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
                        cv2.putText(preview, f'{args.prompt} #{track_id}', (int(x), max(20, int(y) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                cv2.imshow('DARTF FAST webcam - Q to stop', preview)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                stop.set()
    finally:
        stop.set()
        workers[0].join(timeout=2)
        camera.release()
        cv2.destroyAllWindows()
        workers[1].join(timeout=30)
        process.wait()
        workers[2].join(timeout=2)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
