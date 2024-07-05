

param_grid = {
    'sr':           [10000, 22050, 44100],
    'frame_size':   [2**n for n in range(11, 16)],
    'n_coeff':      [10*i for i in range(1,10)],
}