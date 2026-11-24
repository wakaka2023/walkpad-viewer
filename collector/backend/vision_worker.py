import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

from .vision_identity import BREAK_GAP, PersonIdentity, load_encoder

VISION = Path(__file__).resolve().parents[1] / 'runtime' / 'vision'
WEIGHT_HASH = 'a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646'
HEADER = struct.Struct('!Id')
MAX_FRAME = 1024 * 1024
GPU_ID_MODE = 'tracktrack-reid'
COLLECTION_ID_MODE = 'collection-reid'
GPU_ID_MODES = (GPU_ID_MODE, COLLECTION_ID_MODE)
IDENTITY_MODES = ('bytetrack', 'reid', *GPU_ID_MODES)
LIVE_RULES = 'live'
DEFAULT_RULES = 'r13'
LEGACY_RULES = ('r10',)
ACTION_RULES = (DEFAULT_RULES, LIVE_RULES)


def identity_mode():
    mode = os.environ.get('WALKPAD_VISION_ID_MODEL', 'bytetrack').strip().lower()
    if mode not in IDENTITY_MODES:
        raise ValueError(f'不支持的身份模式，可选：{", ".join(IDENTITY_MODES)}')
    return mode


def action_rules():
    mode = os.environ.get('WALKPAD_VISION_ACTION_RULES', DEFAULT_RULES).strip().lower()
    # 未刷新的旧页面仍会传 r10，规则目录已不带版本号，按当前规则跑
    if mode in LEGACY_RULES:
        mode = DEFAULT_RULES
    if mode not in ACTION_RULES:
        raise ValueError(f'动作规则只支持 {", ".join(ACTION_RULES)}')
    return mode


def send(out, message):
    out.write((json.dumps(message, ensure_ascii=True, allow_nan=False) + '\n').encode('ascii'))
    out.flush()


def setup(mode):
    require_gpu = os.environ.get('WALKPAD_VISION_REQUIRE_GPU') == '1'
    if mode in GPU_ID_MODES and not require_gpu:
        raise ValueError('TrackTrack + ReID（GPU）需选择工作站推理')
    weights = VISION / 'yolo26s-pose.pt'
    if not weights.is_file():
        raise ValueError('本机缺少视觉权重 yolo26s-pose.pt')
    with weights.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != WEIGHT_HASH:
            raise ValueError('视觉权重与 v4.9.8 不一致')
    sys.path.insert(0, str(VISION / 'deps'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(VISION / 'config')
    os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '1048576'
    import cv2
    import lap
    import numpy as np
    import torch
    import torchvision
    from ultralytics import YOLO

    cv2.setNumThreads(1)
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        if require_gpu:
            raise ValueError('工作站 CUDA 不可用')
        torch.set_num_threads(2)
    model = YOLO(str(weights))
    if model.task != 'pose':
        raise ValueError('权重不是姿态模型')
    tracker = str(Path(__file__).with_name('vision_bytetrack.yaml'))
    if mode in GPU_ID_MODES:
        from .vision_tracktrack import TrackTrackReID

        tracked = TrackTrackReID(cv2, VISION, device)

        def infer(image):
            result = model.predict(image, device=device, imgsz=960, conf=.10, verbose=False, classes=[0])[0]
            return tracked.update(result)
    else:
        def infer(image):
            result = model.track(image, persist=True, tracker=tracker, device=device,
                                 imgsz=960, conf=.10, verbose=False, classes=[0])[0]
            return result, None

    if require_gpu:
        infer(np.zeros((540, 960, 3), dtype=np.uint8))
    return cv2, np, infer, device


def detections(result, width, height):
    boxes, pose = result.boxes, result.keypoints
    if boxes is None or boxes.id is None or pose is None:
        return []
    ids = boxes.id.cpu().numpy()
    bounds = boxes.xyxy.cpu().numpy()
    points = pose.data.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    if points.shape != (len(ids), 17, 3):
        raise ValueError('姿态模型未返回 17 个关节')
    seen = set()
    output = []
    for j in sorted(range(len(ids)), key=lambda j: -scores[j]):
        if not math.isfinite(ids[j]):
            continue
        identity = int(ids[j])
        x0, y0, x1, y1 = (float(x) for x in bounds[j])
        if identity < 1 or identity in seen or not all(map(math.isfinite, (x0,y0,x1,y1))):
            continue
        box = [max(0., min(1., value)) for value in (x0/width, y0/height, x1/width, y1/height)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        seen.add(identity)
        # Uniform scale preserves angles on portrait and landscape images.
        points[j, :, :2] /= max(width, height)
        output.append((identity, box, points[j]))
    return output


def main():
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    pin = sys.stdin.buffer
    rules = None
    try:
        mode = identity_mode()
        rules = action_rules()
        cv2, np, infer, device = setup(mode)
        identity, pose_gate = None, None
        if mode == COLLECTION_ID_MODE:
            from .vision_session import CollectionIdentity

            identity = CollectionIdentity()
        elif mode == 'reid':
            identity = PersonIdentity(load_encoder(cv2, VISION))
        elif mode == GPU_ID_MODE:
            from .vision_identity import SessionIdentity

            identity = SessionIdentity()
        if mode != COLLECTION_ID_MODE:
            from .vision_session import PoseGate

            pose_gate = PoseGate()
        last_seen = {}
        retention = identity.ttl if identity else 3.
        if rules == LIVE_RULES:
            from .vision_rules_live import LiveRules

            actions = LiveRules(cv2, retention)
        else:
            from .vision_rules_r10 import R10Actions

            actions = R10Actions(cv2, retention)
        ready = {'type': 'ready', 'device': 'GPU' if device == 0 else 'CPU', 'rules': rules, 'identity': mode}
        if mode in GPU_ID_MODES:
            ready['reid_device'] = 'GPU'
        send(out, ready)
        last_seq, last_t = 0, -1.
        while size := pin.read(4):
            length = int.from_bytes(size, 'big')
            if length == 0 and mode == COLLECTION_ID_MODE:
                send(out, identity.review())
                return
            if not HEADER.size < length <= MAX_FRAME:
                raise ValueError('识别帧大小无效')
            packet = pin.read(length)
            if len(packet) != length:
                break
            seq, t = HEADER.unpack_from(packet)
            if seq <= last_seq or not math.isfinite(t) or t <= last_t:
                raise ValueError('识别帧顺序无效')
            image = cv2.imdecode(np.frombuffer(packet, np.uint8, offset=HEADER.size), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('识别帧不是有效图片')
            h, w = image.shape[:2]
            result, features = infer(image)
            state = detections(result, w, h)
            pending_count = 0
            recovered = set()
            if pose_gate is not None:
                state, _, pending_count = pose_gate.frame(t, image, state)
            if mode == COLLECTION_ID_MODE:
                state, recovered, pending_count = identity.resolve(t, image, state, features)
            elif mode == GPU_ID_MODE:
                state, recovered = identity.resolve(t, image, state, [features[key] for key, _, _ in state])
                pending_count += len(identity.pending)
            elif identity is not None:
                state, recovered = identity.resolve(t, image, state)
            last_seen = {pid: at for pid, at in last_seen.items() if t - at <= retention}
            for pid, _, _ in state:
                if t - last_seen.get(pid, t) > BREAK_GAP:
                    recovered.add(pid)
                last_seen[pid] = t
            actions.reset(recovered)
            people = actions.frame(t, image, state)
            send(out, {'type': 'result', 'seq': seq, 't': t, 'people': people, 'pending_count': pending_count})
            last_seq, last_t = seq, t
    except Exception as exc:
        if isinstance(exc, ImportError):
            prefix = '视觉识别依赖不可用：'
            message = prefix + str(exc)
        else:
            message = '本机识别失败：' + str(exc)
        send(out, {'type': 'error', 'message': message[:300]})


if __name__ == '__main__':
    main()
