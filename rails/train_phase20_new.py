from cv2 import threshold
import tqdm
import numpy as np
import torch
from .rails import RAILS
from .datasets import data_loader
from .logger import Logger

import matplotlib.pyplot as plt
import sys
#from PIL import Image
import imageio



def main(args):
    
    rails = RAILS(args)
    data, validation = data_loader('custom_main', args)
    #TODO: implement logger for autoencoder, reconstructions and loss, val_loss, test_loss maybe
    #logger = Logger('carla_train_phase2', args)
    #save_dir = logger.save_dir
    save_dir = "encoder_models"

    if args.resume:
        print ("Loading checkpoint from", args.resume)
        if rails.multi_gpu:
            rails.main_model.module.load_state_dict(torch.load(args.resume))
        else:
            rails.main_model.load_state_dict(torch.load(args.resume))
        start = int(args.resume.split('main_model_')[-1].split('.th')[0])
    else:
        start = 0

    global_it = 0
    for epoch in range(start,start+args.num_epoch):
        cumulative_training_loss = 0
        cumulative_validation_loss = 0
        seg_loss = 0
        rgb_loss = 0

        #Validation
        for wide_rgbs, wide_sems in tqdm.tqdm(validation, desc='Epoch {} Validation'.format(epoch)):
            opt_info_validation = rails.evaluate_autoencoder(wide_rgbs, wide_sems, spds=None)

            cumulative_validation_loss += opt_info_validation.pop("combined_loss")
        
        
        #Training
        for wide_rgbs, wide_sems in tqdm.tqdm(data, desc='Epoch {} Training'.format(epoch)):
            opt_info = rails.train_encoder(wide_rgbs, wide_sems, spds=None)

            cumulative_training_loss += opt_info.pop("combined_loss")
            rgb_loss += opt_info.pop("decoder_loss")
            seg_loss += opt_info.pop("seg_loss")


        wide_rgb = opt_info_validation.pop("wide_rgb")
        decoded_rgb = opt_info_validation.pop("decoded_rgb")
        pred_seg = opt_info_validation.pop("pred_seg")
        gt_seg = opt_info_validation.pop("gt_seg")

        # wide_rgb = opt_info.pop("wide_rgb")
        # decoded_rgb = opt_info.pop("decoded_rgb")
        # pred_seg = opt_info.pop("pred_seg")
        # gt_seg = opt_info.pop("gt_seg")
            
            # if global_it % args.num_per_log == 0:
            #     logger.log_main_info(global_it, opt_info)
        
            # global_it += 1
        print("Epoch training loss: ", cumulative_training_loss/len(data.dataset))
        print("Epoch validation loss: ", cumulative_validation_loss/len(validation.dataset))

        with open("training_losses_ca333000.csv", "a") as training_loss_file:
            training_loss_file.write(f"{epoch}, {cumulative_training_loss/1000000}, {cumulative_validation_loss/15000}\n")

        imageio.imwrite(f"autoencoder_outputs/{epoch}_wideRGB.jpg", wide_rgb)
        imageio.imwrite(f"autoencoder_outputs/{epoch}_decodedRGB.jpg", decoded_rgb)
        imageio.imwrite(f"autoencoder_outputs/{epoch}_gtSEG.jpg", gt_seg)
        imageio.imwrite(f"autoencoder_outputs/{epoch}_predSEG.jpg", pred_seg)
        # Save model
        if (epoch+1) % args.num_per_save == 0:
            save_path = f'{save_dir}/encoder_model_{epoch+1}.th'
            torch.save(rails.encoder_state_dict(), save_path)
            print (f'saved to {save_path}')

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--resume', default=None)
    
    parser.add_argument('--data-dir', default='/lhome/asszewcz/Documents/WorldOnRails/main_data_dir/main_trajs6_converted2')
    parser.add_argument('--config-path', default='/lhome/asszewcz/Documents/WorldOnRails/config_nocrash.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    
    # Training data config
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--num-repeat', type=int, default=4)    # Should be consistent with autoagents/collector_agents/config.yaml

    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epoch', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=3e-5)
    
    parser.add_argument('--num-per-log', type=int, default=100, help='per iter')
    parser.add_argument('--num-per-save', type=int, default=1, help='per epoch')
    
    parser.add_argument('--balanced-cmd', action='store_true')

    args = parser.parse_args()
    main(args)
