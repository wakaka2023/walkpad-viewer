import sys

from annotate_videos import main


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if '--multi-person' in arguments:
        from recognize_multi_actions import main as multi_main
        multi_main([value for value in arguments if value != '--multi-person'])
        raise SystemExit(0)
    if any(value == '--task' or value.startswith('--task=') for value in arguments):
        raise SystemExit('This entry point always recognizes activities; omit --task.')
    if not any(value == '--device' or value.startswith('--device=') for value in arguments):
        arguments += ['--device', 'auto']
    main(arguments + ['--task', 'activities'])
