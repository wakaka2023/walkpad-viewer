from run import main

VIDEO_PATH = r'input/video.mp4'
DATA_PATH = r'input/data.txt'
OUTPUT_DIR = ''
DEVICE = 'auto'
SYNC_MODE = 'common-start'
GRADING_CONFIG = ''
VIDEO_REVIEW = ''
VIDEO_CONFIG = ''

if __name__ == '__main__':
    arguments = ['--video',VIDEO_PATH,'--data',DATA_PATH,'--device',DEVICE,'--sync-mode',SYNC_MODE]
    for option, value in [('--output',OUTPUT_DIR),('--grading-config',GRADING_CONFIG),('--video-review',VIDEO_REVIEW),('--video-config',VIDEO_CONFIG)]:
        if value:
            arguments += [option,value]
    main(arguments)
