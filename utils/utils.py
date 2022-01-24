import os
import shutil
import glob
import torch

def get_most_recent_checkpoint(checkpoint_dir):
    
    checkpoint_paths = [path for path in glob.glob("{}/checkpoint_*".format(checkpoint_dir))]
    lastest_checkpoint=None

    if len(checkpoint_paths) != 0:
        idxes = [int(os.path.basename(path).split('_')[-1]) for path in checkpoint_paths] # [scalar]
        max_idx = max(idxes) # scalar
        lastest_checkpoint = os.path.join(checkpoint_dir, "checkpoint_{}".format(max_idx))        
        print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    else:
        None
        
    return lastest_checkpoint

def save_checkpoint(model, optimizer, iteration, filepath):

    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}/checkpoint_{iteration}")

    torch.save({'iteration': iteration,
                # 'model': (model.module if num_gpus > 1 else model).state_dict(),
                'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()},
                f'{filepath}/checkpoint_{iteration}')

    return True

def load_checkpoint(hparams, model, optimizer):
    
    checkpoint_path = f"{hparams.output_directory}/{hparams.log_directory}"
    checkpoint_path = get_most_recent_checkpoint(checkpoint_path)
    iteration = -1
          
    if checkpoint_path is not None:        
        checkpoint_dict = torch.load(checkpoint_path, map_location='cuda') # , map_location=device)
        model.load_state_dict(checkpoint_dict['model'], strict=False)  
        optimizer.load_state_dict(checkpoint_dict['optimizer']) 
        iteration = checkpoint_dict['iteration'] 
        
        print(f'Loading: {checkpoint_path} // iteration: {iteration} ---------------------------------------')

    return model, optimizer, iteration

def display_result_and_save_tensorboard(writer, result_dict, iteration):

    for key, value in result_dict.items():        
        writer.add_scalar(key, value, iteration)

    return writer

def copy_code(output_directory, log_directory): 

    save_path=f'{output_directory}/{log_directory}/code/'

    npz_path = list()
    for folder in ['model', 'text', 'utils', 'vocoder', '']:

        path_list1 = glob.glob(os.path.join(folder, '*.py'))
        path_list = [path.split('/')[-1] for path in path_list1] # file name
        if not os.path.exists(os.path.join(save_path, folder)):
            os.makedirs(os.path.join(save_path, folder))

        if folder == "text": # 하위폴더 추가
            path_list2 = glob.glob(os.path.join(folder, '*/*.py'))
            path_list += ['/'.join(path.split('/')[-2:]) for path in path_list2] # file name
            folders = set([path.split('/')[-2] for path in path_list])
            for f2 in folders:
                if not os.path.exists(os.path.join(save_path, folder, f2)):
                    os.makedirs(os.path.join(save_path, folder, f2))

        npz_path += [(os.path.join(folder, target), os.path.join(save_path, folder, target)) for target in path_list]
        
    for source_list, target_list in npz_path:
        shutil.copy(source_list, target_list)

    return True

