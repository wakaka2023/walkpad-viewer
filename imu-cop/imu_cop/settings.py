"""One timebase for preparation, learning, file inference and streaming."""
RATE = 50
CONTEXT = 150
WARMUP = 50
STRIDE = 6  # 120 ms: same training endpoint spacing as 25 Hz / stride 3.
ARCHITECTURE = 'action-causal-tcn-50hz-v1'


def validate_checkpoint(checkpoint):
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('This runtime requires a trained 50 Hz checkpoint')
    if checkpoint.get('rate') != RATE or checkpoint.get('context') != CONTEXT:
        raise ValueError('Checkpoint rate/context must be 50 Hz / 150 frames')
