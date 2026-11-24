from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np

from action_rules import ACTIVITY_LABELS, COLORS
from video_to_intervals import display_image


def find_font(path=None):
    if path:
        if not Path(path).is_file():
            raise ValueError('Requested annotation font does not exist.')
        return str(path)
    candidates = ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']
    return next((value for value in candidates if Path(value).is_file()), None)


def render(source, destination, data, frames, segments, events, font_path=None):
    import av
    from PIL import Image, ImageDraw, ImageFont
    width, height = int(data['width']), int(data['height'])
    chosen_font = find_font(font_path)
    chinese = chosen_font is not None
    scale = min(1.5, max(.65, width / 720))
    header, footer = round(158 * scale / 2) * 2, round(94 * scale / 2) * 2
    fonts = {}
    def font(size):
        size = max(10, round(size * scale))
        if size not in fonts:
            fonts[size] = ImageFont.truetype(chosen_font, size) if chosen_font else ImageFont.load_default()
        return fonts[size]
    def text(draw, point, value, size, color):
        draw.text(point, value, font=font(size), fill=color)
    event_map = {row['event_id']: row for row in events}
    duration = frames[-1]['frame_end_s']
    times = np.asarray(data['time_s'], float)
    tb = Fraction(1, 90000)
    count = 0
    source_origin = Fraction(str(float(data.get('first_pts_seconds', 0))))
    with av.open(str(source)) as reader, av.open(str(destination), 'w', options={'movflags': '+faststart'}) as writer:
        video = reader.streams.video[0]
        output = writer.add_stream('libx264', rate=video.average_rate or Fraction(30, 1))
        output.width, output.height, output.pix_fmt = width, height + header + footer, 'yuv420p'
        output.time_base = output.codec_context.time_base = tb
        output.options = {'crf': '21', 'preset': 'fast', 'bf': '0', 'threads': '4'}
        audio_map = {stream.index: writer.add_stream_from_template(stream) for stream in reader.streams.audio}
        first = None
        for packet in reader.demux():
            if packet.stream.type == 'video':
                for frame in packet.decode():
                    absolute = frame.pts * frame.time_base
                    first = absolute if first is None else first
                    time = float(absolute - first)
                    if count >= len(frames) or abs(time - times[count]) > 1 / 90000:
                        raise ValueError('Annotation source PTS differs from the pose cache.')
                    image = Image.fromarray(display_image(frame)[:, :, ::-1])
                    if image.size != (width, height):
                        raise ValueError('Video display geometry differs from the pose cache.')
                    canvas = Image.new('RGB', (width, height + header + footer), '#111A29')
                    canvas.paste(image, (0, header))
                    draw = ImageDraw.Draw(canvas)
                    row = frames[count]
                    segment = segments[row['segment_id'] - 1]
                    kind = row['activity']
                    color = COLORS[kind]
                    pad = round(22 * scale)
                    draw.rectangle((0, 0, max(4, round(7 * scale)), header - 1), fill=color)
                    title = '\u81ea\u52a8\u52a8\u4f5c\u8bc6\u522b\uff5c\u5019\u9009\u5f85\u590d\u6838' if chinese else 'AUTOMATIC ACTION CANDIDATES'
                    text(draw, (pad, 10 * scale), title, 18, '#CDD7E5')
                    text(draw, (width - 221 * scale, 12 * scale), f'{time:05.2f} / {duration:05.2f} s', 18, '#ECF1F8')
                    label = ACTIVITY_LABELS[kind] if chinese else kind.replace('_', ' ').upper()
                    size = 34
                    while size > 17 and draw.textbbox((0, 0), label, font=font(size))[2] > width - 2 * pad:
                        size -= 1
                    text(draw, (pad, 40 * scale), label, size, color)
                    event = event_map.get(row['action_event_id'])
                    if event:
                        phase = ('\u4e0b\u964d' if time <= event['peak_video_s'] else '\u56de\u5347') if chinese else ('DOWN' if time <= event['peak_video_s'] else 'UP')
                        identities = [int(value) for value in segment['event_ids'].split(',') if value]
                        ordinal = identities.index(event['event_id']) + 1
                        detail = f"{ordinal}/{len(identities)}  {phase}   {event['start_video_s']:.2f}\u2013{event['end_video_s']:.2f} s"
                    elif row.get('phase') == 'between_reps':
                        detail = ('\u7ec4\u5185\u56de\u4f4d / \u51c6\u5907\u4e0b\u4e00\u6b21  ' if chinese else 'RECOVERY BETWEEN REPETITIONS  ') + f"{segment['start_video_s']:.2f}\u2013{segment['end_video_s']:.2f} s"
                    else:
                        detail = (f"\u7b2c {row['segment_id']}/{len(segments)} \u6bb5   " if chinese else f"SEGMENT {row['segment_id']}/{len(segments)}   ") + f"{segment['start_video_s']:.2f}\u2013{segment['end_video_s']:.2f} s"
                    text(draw, (pad, 91 * scale), detail, 21, '#F1F5FA')
                    note = ('\u8bc1\u636e\u6709\u9650\uff1a\u5c40\u90e8\u53ef\u89c1\u6216\u8bed\u4e49\u672a\u786e\u8ba4' if row['evidence_level'] == 'limited' else '\u89c6\u89c9\u89c4\u5219\u81ea\u52a8\u5206\u6bb5\uff0c\u8fb9\u754c\u9700\u590d\u6838') if chinese else 'UNREVIEWED VISUAL RULE PROPOSAL'
                    if row.get('context_inferred'):
                        note = '\u77ed\u6682\u8bc1\u636e\u4e0d\u5168\uff1a\u4f9d\u636e\u76f8\u90bb\u753b\u9762\u4fdd\u6301\u52a8\u4f5c' if chinese else 'BRIEF GAP: ACTIVITY INFERRED FROM TEMPORAL CONTEXT'
                    text(draw, (pad, 126 * scale), note, 17, '#B9C7D8')
                    y = header + height
                    caution = '\u8d70/\u8dd1\u4e0d\u786e\u5b9a\u65f6\u4fdd\u7559\u6807\u8bb0\uff1b\u672a\u89c6\u4e3a\u5df2\u9a8c\u8bc1\u771f\u503c' if chinese else 'UNRESOLVED ACTIONS RETAIN CANDIDATE LABELS'
                    text(draw, (pad, y + 10 * scale), caution, 18, '#FFD166')
                    left, right, bar_y = pad, width - pad, y + round(48 * scale)
                    for item in segments:
                        x0 = left + (right - left) * item['start_video_s'] / duration
                        x1 = left + (right - left) * item['end_video_s'] / duration
                        draw.rectangle((x0, bar_y, max(x0 + 1, x1), bar_y + 12 * scale), fill=COLORS[item['activity']])
                    x = left + (right - left) * time / duration
                    draw.line((x, bar_y - 5 * scale, x, bar_y + 18 * scale), fill='white', width=max(2, round(3 * scale)))
                    rendered = av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')
                    rendered.pts, rendered.time_base = round(time * 90000), tb
                    for encoded in output.encode(rendered):
                        writer.mux(encoded)
                    count += 1
                    if count % 600 == 0:
                        print('ACTION_VIDEO_FRAMES {}/{}'.format(count, len(frames)), flush=True)
            elif packet.stream.index in audio_map and packet.dts is not None:
                offset = round(source_origin / packet.time_base)
                if packet.pts is not None:
                    packet.pts -= offset
                packet.dts -= offset
                packet.stream = audio_map[packet.stream.index]
                writer.mux(packet)
        for packet in output.encode():
            writer.mux(packet)
    if count != len(frames):
        raise ValueError('Annotated source count differs from the pose cache.')


def audio_digest(path):
    import av
    digest, count = hashlib.sha256(), 0
    with av.open(str(path)) as reader:
        if not reader.streams.audio:
            return None, 0
        for packet in reader.demux(audio=0):
            if packet.size:
                digest.update(bytes(packet))
                count += 1
    return digest.hexdigest(), count


def verify(source, output, data):
    import av
    expected = np.asarray(data['time_s'], float)
    with av.open(str(output)) as reader:
        stream = reader.streams.video[0]
        dimensions = [stream.width, stream.height]
        actual = np.array([float(frame.pts * frame.time_base) for frame in reader.decode(video=0)])
    if len(actual) != len(expected):
        raise ValueError('Encoded video count differs from the source.')
    error = float(np.max(np.abs(actual - actual[0] - expected)))
    if error > 1 / 90000 + 1e-9:
        raise ValueError('Encoded video PTS differs from the source.')
    source_audio, output_audio = audio_digest(source), audio_digest(output)
    if source_audio != output_audio:
        raise ValueError('Audio packets changed during annotated export.')
    return dict(frame_count=len(actual), max_pts_error_s=error, audio_packets=output_audio[1],
                audio_sha256=output_audio[0], original_audio_preserved=True,
                source_display_region_preserved=True, output_dimensions=dimensions)
