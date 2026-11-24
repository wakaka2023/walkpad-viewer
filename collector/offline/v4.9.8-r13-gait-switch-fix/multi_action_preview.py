from fractions import Fraction
from functools import lru_cache
import hashlib

import numpy as np

from action_preview import find_font
from multi_track import oriented_image

PALETTE = ('#39D6FF', '#FFAD46', '#BBA8FF', '#60E6AA', '#FF81AE', '#E9EF73', '#F3F6FF', '#FB8864')
LIMBS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]
from multi_labels import LABELS
PHASES = dict(down='\u4e0b\u964d', up='\u56de\u5347', between_reps='\u7ec4\u5185\u56de\u4f4d')


def intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_people(canvas, detections, rows, *, time, keypoint_conf, font, use_chinese, text_bbox=None):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    draw.rectangle((0, 0, w, 54), fill='#121D30')
    status = f'\u591a\u4eba\u52a8\u4f5c\u6807\u6ce8  |  \u5f53\u524d {len(rows)} \u4eba  |  {time:.2f} s' if use_chinese else f'MULTI-PERSON ACTIONS  |  {len(rows)} tracks  |  {time:.2f} s'
    draw.text((15, 13), status, fill='#F1F6FF', font=font)
    pending = []
    if len(detections) != len(rows):
        raise ValueError('Overlay needs one label per detection.')
    for (identity, coords, pose), row in zip(detections, rows):
        if row['track_id'] != identity:
            raise ValueError('Overlay attempted to pair a track with another ID.')
        color = PALETTE[(identity - 1) % len(PALETTE)]
        box = tuple(round(float(value)) for value in coords)
        x1, y1, x2, y2 = box
        y1, y2 = y1 + 54, y2 + 54
        box = (x1, y1, x2, y2)
        draw.rectangle(box, outline=color, width=3)
        pose = np.asarray(pose)
        valid = np.isfinite(pose).all(axis=1) & (pose[:, 2] >= keypoint_conf)
        if row.get('gait_evidence','').startswith('cropped_'):
            valid[:] = False
        xy = [tuple(x) for x in np.nan_to_num(pose[:, :2]) + (0, 54)]
        for i, j in LIMBS:
            if valid[i] and valid[j]:
                draw.line((xy[i], xy[j]), fill=color, width=3)
        event = row.get('action_event_id', 0)
        phase = row.get('phase', '')
        kind = row['activity'] if row['activity'] in LABELS else 'unknown'
        if use_chinese:
            action = row.get('annotation', LABELS[kind])
        elif row.get('activity_status') == 'candidate':
            action = ' / '.join(x.replace('_',' ').upper()+' CANDIDATE' for x in row['action_candidates'].split(';'))
        else:
            action = 'OTHER' if kind == 'unknown' else kind.replace('_',' ').upper()
        text = f'ID {identity}  {action}'
        if event:
            text += f' #{event}'
        if phase:
            text += ' ' + (PHASES.get(phase, phase) if use_chinese else phase.upper())
        pending.append((box, color, text))
    placed = []
    for (x1, y1, x2, y2), color, text in sorted(pending, key=lambda x: x[0][1]):
        metrics = text_bbox(text) if text_bbox is not None else draw.textbbox((0, 0), text, font=font)
        tw, th = min(w, metrics[2] + 16), max(31, metrics[3] + 8)
        x = max(0, min(w - tw, x1))
        candidates = [(x, y1-th-3), (x, y1+3), (x, y2+3), (x, y2-th-3),
                      (min(w-tw, x2+3), y1), (max(0,x1-tw-3), y1)]
        candidates += [(x, yy) for yy in range(55, max(56, h-th), th+2)]
        options = [(xx, yy, xx+tw, yy+th) for xx, yy in candidates if 54 <= yy <= h-th]
        panel = next((r for r in options if not any(intersects(r, q) for q in placed)),
                     options[0] if options else (x, h-th, x+tw, h))
        placed.append(panel)
        draw.line((panel[0], panel[1], x1, y1), fill=color, width=2)
        draw.rectangle(panel, fill='#101926', outline=color, width=2)
        draw.text((panel[0]+7, panel[1]+1), text, font=font, fill='#FFFFFF')


def render(source, destination, data, frame_rows, *, keypoint_conf, font_path):
    import av
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    width, height = int(data['width']), int(data['height'])
    times = data['time_s']
    lookup = {(row['video_frame']-1, row['track_id']): row for row in frame_rows if row['track_state'] in ('tracked','image_supported_continuation')}
    font_name = find_font(font_path)
    font = ImageFont.truetype(font_name, min(29, max(16, width//42))) if font_name else ImageFont.load_default()
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))

    @lru_cache(maxsize=512)
    def text_bbox(text):
        return measure.textbbox((0, 0), text, font=font)

    rows_by_frame = [[] for _ in times]
    for j, (i, identity) in enumerate(zip(data['frame_index'], data['track_id'])):
        if (i, identity) in lookup:
            rows_by_frame[i].append((int(identity), data['boxes'][j], data['keypoints'][j]))
    tb = Fraction(1, 90000)
    count = 0
    source_origin = Fraction(str(float(data['first_pts_seconds'])))
    with av.open(str(source)) as reader, av.open(str(destination), 'w', options={'movflags': '+faststart'}) as writer:
        v = reader.streams.video[0]
        output = writer.add_stream('libx264', rate=v.average_rate or Fraction(30, 1))
        output.width, output.height, output.pix_fmt = width + width % 2, height + height % 2 + 54, 'yuv420p'
        rgb = np.empty((output.height, output.width, 3), dtype=np.uint8)
        rgb[:] = (18, 29, 48)
        region = rgb[54:54+height, :width]
        output.time_base = output.codec_context.time_base = tb
        output.options = {'crf': '21', 'preset': 'veryfast', 'bf': '0', 'threads': '4'}
        audio_map = {stream.index: writer.add_stream_from_template(stream) for stream in reader.streams.audio}
        first = None
        for packet in reader.demux([v, *reader.streams.audio]):
            if packet.stream.index == v.index:
                for frame in packet.decode():
                    absolute = frame.pts * frame.time_base
                    first = absolute if first is None else first
                    relative = absolute - first
                    time = float(relative)
                    if count >= len(times) or abs(time - times[count]) > 1 / 90000:
                        raise ValueError('Annotated source timeline differs from tracking cache.')
                    image = oriented_image(frame, int(data['rotate']))
                    if image.shape[:2] != (height, width):
                        raise ValueError('Annotated source geometry differs from tracking cache.')
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=region)
                    canvas = Image.fromarray(rgb)
                    dets = rows_by_frame[count]
                    draw_people(canvas, dets, [lookup[count, item[0]] for item in dets], time=time,
                                keypoint_conf=keypoint_conf, font=font, use_chinese=bool(font_name), text_bbox=text_bbox)
                    img = av.VideoFrame.from_image(canvas)
                    img.pts, img.time_base = round(relative / tb), tb
                    for encoded in output.encode(img):
                        writer.mux(encoded)
                    count += 1
                    if count % 600 == 0:
                        print(f'RENDER frame={count}/{len(times)}', flush=True)
            elif packet.dts is not None and packet.size:
                offset = round(source_origin / packet.time_base)
                if packet.pts is not None:
                    packet.pts -= offset
                packet.dts -= offset
                packet.stream = audio_map[packet.stream.index]
                writer.mux(packet)
        for encoded in output.encode():
            writer.mux(encoded)
    if count != len(times):
        raise ValueError('Annotated source count differs from the tracking cache.')


def stream_signatures(path, *, video=False):
    import av
    timestamps = []
    dimensions = None
    with av.open(str(path)) as reader:
        streams = list(reader.streams.audio)
        if video:
            stream = reader.streams.video[0]
            dimensions = [stream.width, stream.height]
        if not streams and not video:
            return timestamps, dimensions, []
        order = {stream.index: j for j, stream in enumerate(streams)}
        hashes, counts = [hashlib.sha256() for _ in streams], [0]*len(streams)
        for packet in reader.demux([stream, *streams] if video else streams):
            j = order.get(packet.stream.index)
            if j is None:
                timestamps.extend(float(frame.pts * frame.time_base) for frame in packet.decode())
            elif packet.size:
                hashes[j].update(bytes(packet))
                counts[j] += 1
    return timestamps, dimensions, [dict(packets=n, sha256=h.hexdigest()) for n, h in zip(counts, hashes)]


def audio_stream_signatures(path):
    return stream_signatures(path)[2]


def verify(source, output, data):
    expected = np.asarray(data['time_s'], float)
    timestamps, dimensions, annotated = stream_signatures(output, video=True)
    if len(timestamps) != len(expected):
        raise ValueError('Encoded video count differs from the source.')
    actual = np.asarray(timestamps)
    error = float(np.max(np.abs(actual - actual[0] - expected)))
    if error > 1 / 90000 + 1e-9:
        raise ValueError('Encoded video PTS differs from the source.')
    if dimensions != [int(data['width']) + int(data['width']) % 2,
                      int(data['height']) + int(data['height']) % 2 + 54]:
        raise ValueError('Encoded video dimensions differ from the overlay geometry.')
    if audio_stream_signatures(source) != annotated:
        raise ValueError('One or more audio streams changed during annotated export.')
    first = annotated[0] if annotated else dict(packets=0, sha256=None)
    return dict(frame_count=len(actual), max_pts_error_s=error, audio_packets=first['packets'],
                audio_sha256=first['sha256'], original_audio_preserved=True,
                source_display_region_preserved=True, output_dimensions=dimensions,
                audio_streams=annotated, clockwise_correction_deg=int(data['rotate']),
                simultaneous_track_overlay=True)
