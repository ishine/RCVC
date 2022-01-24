import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

def save_figure_to_numpy(fig):

    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
    
def plot_alignment_to_numpy(alignment, iteration, width, height, text_sequence=None, text=None):
    
    audio_len, char_len = alignment.shape
    
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)

    if text_sequence is not None:   
        A = [x if x!= '0' else '' for x in text_sequence] 
        plt.xticks(range(char_len), A)

    plt.ylabel('Decoder timestep')
    plt.xlabel('Encoder timestep')
    plt.title(text)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()

    return data

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        writer = TTSWriter(logging_path)
        print(f'The experiment {logging_path} already exists!')
    else:
        os.makedirs(logging_path)
        writer = TTSWriter(logging_path)
            
    return writer

class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)

    def add_specs(self, data, global_step, phase):

        fig, axes = plt.subplots(len(data), 1, figsize=(12,9))

        for i, data in enumerate(data):
            axes[i].xaxis.set_visible(False)
            axes[i].yaxis.set_visible(False)
            axes[i].imshow(data, origin='lower', aspect='auto')

        self.add_figure(f'{phase}_melspec', fig, global_step)
        plt.close()

    def add_alignments(self, alignment, global_step, phase):

        fig = plt.plot(figsize=(20,10))
        plt.imshow(alignment, origin='lower', aspect='auto')
        self.add_figure(f'{phase}_alignments', fig, global_step)
        plt.close()
    
    def add_alignments2(self, alignment, global_step, phase, width, height, text_sequence=None, text=None):

        self.add_image(
            phase,
            plot_alignment_to_numpy(alignment, global_step, width, height, text_sequence, text),
            global_step, dataformats='HWC')