import numpy as np
from .video_to_intervals import display_image

def render_preview(video, out, data, frames, intervals, segments, *, max_height=None, preset='ultrafast', frame_overlay=None):
    import av
    import cv2
    from fractions import Fraction

    t = np.asarray(data["time_s"])
    width, height = int(data['width']), int(data['height'])
    source_shape = (height, width)
    ratio = min(1., max_height/height) if max_height else 1.
    width, height = max(2, round(width*ratio)//2*2), max(2, round(height*ratio)//2*2)
    colors = {'squat': (40, 210, 80), 'jump': (50, 160, 255),
              'walking': (235, 185, 50), 'standing': (175, 175, 175), 'unknown': (90,90,90)}
    tb = Fraction(1, 90000)
    writer = av.open(str(out), 'w')
    stream = writer.add_stream('libx264', rate=Fraction(30,1))
    stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
    stream.time_base = tb
    stream.codec_context.time_base = tb
    stream.options = {'crf': '22', 'preset': preset, 'bf': '0'}
    count = 0
    first_source_time = None
    def put(im, text, y, color=(255,255,255), size=.65):
        cv2.putText(im, text, (16,y), cv2.FONT_HERSHEY_SIMPLEX,size,color,2,cv2.LINE_AA)
    try:
        with av.open(str(video)) as source:
            for index, frame in enumerate(source.decode(video=0)):
                if index >= len(frames):
                    raise ValueError("Preview video has more frames than the pose cache.")
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Preview source has no presentation timestamps.')
                source_time = frame.pts*frame.time_base
                if first_source_time is None:
                    first_source_time = source_time
                if abs(float(source_time-first_source_time)-t[index]) > 1/90000:
                    raise ValueError('Source PTS differs from the pose cache at frame {}'.format(index+1))
                image = display_image(frame)
                if image.shape[:2] != source_shape:
                    raise ValueError('Source display dimensions differ from pose cache.')
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                sx, sy = width/source_shape[1], height/source_shape[0]
                row = frames[index]
                time = row['video_time_s']
                activity = row.get('activity','unknown')
                color = colors[activity]
                cv2.rectangle(image,(0,0),(width,168),(20,20,20),-1)
                cv2.rectangle(image,(0,0),(width-1,height-1),color,7)
                put(image,'SOURCE F{}/{}    {:.3f} s'.format(index+1,len(frames),time),30)
                title=activity.upper()
                event = next((e for e in intervals if e['rep_id']==row['rep_id']),None)
                if event:
                    phase = row.get('movement_phase') or ('DOWN' if time <= event['bottom_video_s'] else 'UP')
                    title='SQUAT #{} / {}   {}'.format(row['rep_id'],len(intervals),phase)
                put(image,title,65,color,.9)
                recent=[(e,kind) for e in intervals for kind in ('start','end')
                        if 0 <= time-e[kind+'_video_s'] < .4]
                if recent:
                    e,kind=recent[-1]
                    marker_frame=e['start_video_frame'] if kind=='start' else e['end_video_frame']+1
                    put(image,'{} #{}  F{}  {:.3f}s{}'.format(kind.upper(),e['rep_id'],marker_frame,
                        e[kind+'_video_s'],' (exclusive)' if kind=='end' else ''),103,(0,255,255),.7)
                if event:
                    put(image,'Included frames: {} - {}'.format(event['start_video_frame'],event['end_video_frame']),133,color,.6)
                else:
                    put(image,'Video-only rules / boundaries need review',133,(190,190,190),.55)
                mode='SIDE' if any('side_' in str(e.get('body_boundary_status','')) for e in intervals) else 'FRONT'
                quality=' / CHECK REFERENCE' if event and ('turnaround' in str(event.get('before_reference_quality','')) or 'turnaround' in str(event.get('after_reference_quality','')) or 'failed' in str(event.get('body_boundary_status',''))) else ''
                if event and event.get('side_onset_needs_review'):quality+=' / CHECK ONSET'
                if event and event.get('body_boundary_needs_review'):quality+=' / CHECK BOUNDARY'
                put(image,mode+' / TORSO + LEGS'+quality,157,(170,170,170),.45)
                for j in range(17):
                    x, y, c = data["keypoints"][index, j]
                    if np.isfinite([x, y, c]).all() and c >= 0.5:
                        cv2.circle(image, (int(x*sx), int(y*sy)), 4, (0, 255, 0), -1)
                for a,b in [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
                    points = data['keypoints'][index,[a,b]]
                    if np.isfinite(points).all() and (points[:,2]>=.5).all():
                        cv2.line(image,tuple((points[0,:2]*[sx,sy]).astype(int)),tuple((points[1,:2]*[sx,sy]).astype(int)),color,3)
                y = height-45
                cv2.rectangle(image,(10,y-20),(width-10,height-10),(20,20,20),-1)
                for segment in segments:
                    a = 15+round((width-30)*segment['start_video_s']/frames[-1]['frame_end_s'])
                    b = 15+round((width-30)*segment['end_video_s']/frames[-1]['frame_end_s'])
                    cv2.rectangle(image,(a,y),(max(a+1,b),y+20),colors[segment['activity']],-1)
                x=15+round((width-30)*time/frames[-1]['frame_end_s'])
                cv2.line(image,(x,y-8),(x,y+26),(255,255,255),3)
                if frame_overlay is not None:
                    image = frame_overlay(image, row, event)
                output_frame=av.VideoFrame.from_ndarray(image,format='bgr24')
                output_frame.pts=int(round(time*90000))
                output_frame.time_base=tb
                for packet in stream.encode(output_frame):
                    writer.mux(packet)
                count += 1
        if count != len(frames):
            raise ValueError("Preview video frame count differs from the pose cache.")
        for packet in stream.encode():
            writer.mux(packet)
    finally:
        writer.close()
