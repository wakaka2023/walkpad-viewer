import numpy as np

from .video_to_intervals import display_image


def oriented_image(frame, rotate):
    return np.ascontiguousarray(np.rot90(display_image(frame), -(rotate // 90)))


def read_detections(result, width, height):
    if result.boxes is None or not len(result.boxes):
        return np.empty((0, 4), np.float32), []
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    boxes = np.clip(boxes, 0, [width - 1, height - 1, width - 1, height - 1]).astype(np.float32)
    usable = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if result.boxes.id is None:
        return boxes[usable], []
    ids = result.boxes.id.cpu().numpy().astype(np.int64)
    scores = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
    if keypoints is None or keypoints.shape != (len(boxes), 17, 3) or len(ids) != len(boxes):
        raise ValueError('Tracked result does not contain matching COCO 17-keypoint poses.')
    order = np.argsort(-scores, kind='stable')
    seen, tracked = set(), []
    for i in order:
        identity = int(ids[i])
        if not usable[i] or identity <= 0 or identity in seen:
            continue
        seen.add(identity)
        tracked.append((identity, boxes[i], float(scores[i]), keypoints[i].astype(np.float32)))
    return boxes[usable], tracked


def extract(source, destination, model_path, *, imgsz, conf, device, tracker, rotate):
    import av
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != 'pose':
        raise ValueError('Multitrack action recognition requires a COCO pose model.')
    times, records, mask_boxes, mask_offsets = [], [], [], [0]
    first, shape, end = None, None, None
    with av.open(str(source)) as reader:
        for index, frame in enumerate(reader.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Source video must have frame PTS.')
            absolute = frame.pts * frame.time_base
            first = absolute if first is None else first
            time = float(absolute - first)
            if times and time <= times[-1]:
                raise ValueError('Source video PTS must increase strictly.')
            times.append(time)
            if frame.duration:
                end = time + float(frame.duration * frame.time_base)
            else:
                end = None
            image = oriented_image(frame, rotate)
            shape = image.shape[:2] if shape is None else shape
            if image.shape[:2] != shape:
                raise ValueError('Source display dimensions change within the video.')
            result = model.track(image, persist=True, tracker=tracker, device=device, imgsz=imgsz,
                                 conf=conf, verbose=False, classes=[0])[0]
            regions, tracked = read_detections(result, shape[1], shape[0])
            mask_boxes.extend(regions)
            mask_offsets.append(len(mask_boxes))
            records.extend((index, *item) for item in tracked)
            if index % 300 == 0:
                print(f'TRACK frame={index + 1} time={time:.2f}s ids={[item[0] for item in tracked]}', flush=True)
    if len(times) < 3:
        raise ValueError('Video needs at least three frames.')
    end = end if end is not None and end > times[-1] else times[-1] + float(np.median(np.diff(times)))
    values = dict(time_s=np.asarray(times), frame_end_s=np.r_[times[1:], end],
                  first_pts_seconds=float(first), width=shape[1], height=shape[0],
                  frame_index=np.asarray([r[0] for r in records], dtype=np.int32),
                  track_id=np.asarray([r[1] for r in records], dtype=np.int64),
                  boxes=np.asarray([r[2] for r in records], dtype=np.float32).reshape(-1, 4),
                  confidence=np.asarray([r[3] for r in records], dtype=np.float32),
                  keypoints=np.asarray([r[4] for r in records], dtype=np.float32).reshape(-1, 17, 3),
                  mask_boxes=np.asarray(mask_boxes, dtype=np.float32).reshape(-1, 4),
                  mask_offsets=np.asarray(mask_offsets, dtype=np.int64), rotate=rotate)
    np.savez_compressed(destination, **values)
    return values
