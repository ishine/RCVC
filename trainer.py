import warnings
import os, sys
import argparse
warnings.filterwarnings("ignore")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# model
import torch
import model.hparams as hp
from model.model import Model
from model.loss import LossFunction
from model.train import train
from model.validation import validation
from online_inference import run_online_inference

# utils
from utils.utils import save_checkpoint, load_checkpoint, copy_code
from utils.writer import get_writer
from utils.data_utils import prepare_dataloaders
from utils.scheduler import ScheduledOptimizer

def main():

    # Load model
    model = Model(hp).cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, betas=[0.9, 0.98])
    model, optimizer, iteration = load_checkpoint(hp, model, optimizer)
    scaler = torch.cuda.amp.GradScaler()     
    scheduler = ScheduledOptimizer(optimizer, hp.lr, 4000, iteration)

    # criterion
    criterion = LossFunction(hp)

    # loader
    train_loader, val_loader = prepare_dataloaders(hp, hp.num_workers)    
    epoch_offset = max(0, int(iteration / len(train_loader)))

    # helper function
    writer = get_writer(hp.output_directory, f'{hp.log_directory}')
    copy_code(hp.output_directory, f'{hp.log_directory}')

    print(f'Model training start!!! {hp.log_directory}')
    for epoch in range(epoch_offset, 10000):

        print("Epoch: {}, lr: {}".format(epoch + 1, scheduler.get_learning_rate()))
        for i, batch in enumerate(train_loader):

            iteration += 1
            model.train()
            train(model, optimizer, scheduler, criterion, writer, iteration, batch, scaler)   

            if iteration % hp.iters_per_checkpoint == 0:    
                save_checkpoint(model, optimizer, iteration, filepath=f'{hp.output_directory}/{hp.log_directory}')

            if iteration % hp.iters_per_validation == 0:
                validation(model, criterion, writer, val_loader, iteration)  

            if iteration % hp.iters_per_online_inference == 0:
                run_online_inference(hp, model, writer, iteration)
            
            if iteration == hp.stop_iteration:
                break

        if iteration == hp.stop_iteration:
            break

    print(f'Training finish!!!')    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='VCTK') # VCTK, NIKL
    parser.add_argument('--log_dir', default='RCVC_VCTK_test01')  # RCVC_VCTK
    a = parser.parse_args()

    hp.dataset = a.dataset
    hp.log_directory = a.log_dir

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    
    main()