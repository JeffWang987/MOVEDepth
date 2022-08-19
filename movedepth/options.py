import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MOVEDepth options")

        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--depth_binning",
                                 help="defines how the depth bins are constructed for the cost"
                                      "volume. 'linear' is uniformly sampled in depth space,"
                                      "'inverse' is uniformly sampled in inverse depth space",
                                 type=str,
                                 choices=['linear', 'inverse'],
                                 default='linear'),
        self.parser.add_argument("--num_depth_bins",
                                 type=int,
                                 default=16)
        self.parser.add_argument("--ztrans_start_epc",
                                 type=int,
                                 default=8)
        self.parser.add_argument("--depth_bin_fac",
                                 type=float,
                                 default=0.3)
        self.parser.add_argument("--ssim_lw",
                                 type=float,
                                 default=0.85)
        self.parser.add_argument("--split1",
                                 type=float,
                                 default=0.333)
        self.parser.add_argument("--split2",
                                 type=float,
                                 default=0.666)
        self.parser.add_argument("--mask_lw",
                                 type=float,
                                 default=10)
        self.parser.add_argument("--photo_conf",
                                 type=float,
                                 default=0.2)
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 )
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--matching_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1])
        self.parser.add_argument("--casbins",
                                 nargs="+",
                                 type=int,
                                 default=[8, 4, 4])
        self.parser.add_argument("--casfac",
                                 nargs="+",
                                 type=float,
                                 default=[0.5, 0.25, 0.125])
        self.parser.add_argument("--casch",
                                 nargs="+",
                                 type=int,
                                 default=[8, 4, 4])
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--res_arch",
                                 type=int,
                                 default=18)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--pytorch_random_seed",
                                 default=None,
                                 type=int)
        self.parser.add_argument("--update_range_epoch",
                                 type=int,
                                 default=0)
        self.parser.add_argument("--lr_fac",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--enable_mvs_pose_grad",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument('--use_future_frame',
                                 action='store_true',
                                 help='If set, will also use a future frame in time for matching.')
        self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)
        self.parser.add_argument("--disable_motion_masking",
                                 help="If set, will not apply consistency loss in regions where"
                                      "the cost volume is deemed untrustworthy",
                                 action="store_true")
        self.parser.add_argument("--disable_edge_masking",
                                 help="If set, will not apply edge mask in cost volume",
                                 action="store_true")
        self.parser.add_argument("--no_matching_augmentation",
                                 action='store_true',
                                 help="If set, will not apply static camera augmentation or "
                                      "zero cost volume augmentation during training")
        self.parser.add_argument("--group_cor",
                                 action='store_true',
                                 help="If set, use group correlation instead of abs difference to aggregate cost volumes.")
        self.parser.add_argument("--mvs_norm",
                                 action='store_true',
                                 help="If set, norm the neighbour depth bin before regression.")  
        self.parser.add_argument("--conv3d",
                                 action='store_true',
                                 help="If set, use 3d conv to aggregate cost volumes.")              
        self.parser.add_argument("--mono_prior",
                                 action='store_true',
                                 help="If set, use mono depth as prior for mvs.")      
        self.parser.add_argument("--reg3d_c",
                                 type=int,
                                 default=16)    
        self.parser.add_argument("--preconv",
                                 action="store_true")         
        self.parser.add_argument("--log",
                                 action="store_true")  
        self.parser.add_argument("--fix_scale",
                                 action="store_true")
                         
        self.parser.add_argument("--prior_scale",
                                 type=int,
                                 default=2)
        self.parser.add_argument("--norm_radius",
                                 type=int,
                                 default=1)
        self.parser.add_argument("--mvs_cascade",
                                 action="store_true")
        self.parser.add_argument("--mvs_raft",
                                 action="store_true")
        self.parser.add_argument("--schedule_type",
                                 type=str,
                                 default='inverse')
        self.parser.add_argument("--iter_stages",
                                 type=int,
                                 default=4)
        self.parser.add_argument("--iter_bins",
                                 type=int,
                                 default=8)
        self.parser.add_argument("--z_scale",
                                 type=float,
                                 default=30)
        self.parser.add_argument("--dist_thres",
                                 type=float,
                                 default=0)
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--mono_weights_folder",
                                 type=str)
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose, reg3d", "mono_depth", "mono_encoder"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--save_intermediate_models",
                                 help="if set, save the model each time we log to tensorboard",
                                 action='store_true')

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--zero_cost_volume",
                                 action="store_true",
                                 help="If set, during evaluation all poses will be set to 0, and "
                                      "so we will evaluate the model in single frame mode")
        self.parser.add_argument('--static_camera',
                                 action='store_true',
                                 help='If set, during evaluation the current frame will also be'
                                      'used as the lookup frame, to simulate a static camera')
        self.parser.add_argument('--eval_teacher',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        self.parser.add_argument('--convex_up',
                                 action='store_true')
        self.parser.add_argument('--load_pose',
                                 action='store_true')
        self.parser.add_argument('--mask_mvs_conf',
                                 action='store_true')
        self.parser.add_argument('--mask_mvs_dist',
                                 action='store_true')
        self.parser.add_argument('--mask_mvs_geo',
                                 action='store_true')
        self.parser.add_argument('--mask_mvs_auto',
                                 action='store_true')      
        self.parser.add_argument('--mvs_smooth_loss',
                                 action='store_true')  
        self.parser.add_argument('--dcn',
                                 action='store_true')   
        self.parser.add_argument("--pixel_thres",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--depth_thres",
                                 type=float,
                                 default=0.1)
        self.parser.add_argument("--freeze_fuse_epc",
                                 type=int,
                                 default=0)
        self.parser.add_argument('--train_motion_only',
                         action='store_true')  
             
        # DDP
        self.parser.add_argument("--local_rank", default=0,type=int)
        self.parser.add_argument("--ddp", 
                                  help="Use DistributedDataParallel",
                                  action="store_true")
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
