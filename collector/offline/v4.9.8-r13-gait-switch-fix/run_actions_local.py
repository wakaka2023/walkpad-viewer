from annotate_videos import main


INPUT_PATH = r'input/videos'
OUTPUT_DIR = r'results_actions'
DEVICE = 'auto'
RESUME = False


if __name__ == '__main__':
    arguments = [INPUT_PATH, '--output', OUTPUT_DIR, '--task', 'activities', '--device', DEVICE]
    if RESUME:
        arguments.append('--resume')
    main(arguments)
