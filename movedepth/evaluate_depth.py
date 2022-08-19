from csv import writer
import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from movedepth.utils import readlines
from movedepth.options import MonodepthOptions
from movedepth import datasets, networks
from movedepth.layers import transformation_from_parameters, disp_to_depth, schedule_depth_rangev2, BackprojectDepth, Project3D, convex_upsample_layer, generate_costvol, localmax, \
    schedule_depth_range_zv2, entropy
import torch.nn.functional as F
from tqdm import tqdm
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_fuse_errors(gt, pred1, pred2):
    """Computation of error metrics between predicted and ground truth depths
    """
    mask1 = np.abs(gt-pred1) < np.abs(pred2-gt)
    # mask1 = pred1 < 10
    # print(mask1.sum().astype(np.float16)/mask1.reshape(-1).shape[0])
    pred = mask1*pred1 + ~mask1*pred2
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
        
    print("Using ckpt: {}".format(opt.load_weights_folder))
    backprojector = BackprojectDepth(batch_size=opt.num_depth_bins,
                                        height=opt.height//(2**opt.prior_scale),
                                        width=opt.width//(2**opt.prior_scale))
    projector = Project3D(batch_size=opt.num_depth_bins,
                                    height=opt.height//(2**opt.prior_scale),
                                    width=opt.width//(2**opt.prior_scale))
    backprojector.cuda()
    projector.cuda()
    frames_to_load = opt.matching_ids
    HEIGHT, WIDTH = opt.height, opt.width

    img_ext = '.png' if opt.png else '.jpg'
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    filenames = readlines(os.path.dirname(__file__) + "/splits/"+ opt.eval_split + "/test_files.txt")


    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                        HEIGHT, WIDTH,
                                        frames_to_load, 4,
                                        is_train=False,
                                        img_ext=img_ext)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)



    # setup models
    # mono encoder
    mono_encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
    mono_encoder_dict = torch.load(mono_encoder_path)
    mono_encoder = networks.ResnetEncoder(num_layers=opt.res_arch, pretrained=False)
    mono_encoder.load_state_dict(mono_encoder_dict, strict=True)
    mono_encoder.eval()
    mono_encoder.cuda()

    # mono decoder
    mono_decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
    mono_depth_decoder_dict = torch.load(mono_decoder_path)
    mono_depth_decoder = networks.DepthDecoder(mono_encoder.num_ch_enc, match_conv=False, ddv=False, discret=None, mono_conf=False,
                                                 mono_bins=None)
    mono_depth_decoder.load_state_dict(mono_depth_decoder_dict, strict=True)
    mono_depth_decoder.eval()
    mono_depth_decoder.cuda()

    # pose enc
    pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
    pose_enc = networks.ResnetEncoder(opt.res_arch, False, num_input_images=2)
    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_enc.eval()
    pose_enc.cuda()

    # pose dec
    pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)
    pose_dec.eval()
    pose_dec.cuda()

    # mvs encoder
    mvs_encoder_dict = torch.load(os.path.join(opt.load_weights_folder, "mvs_encoder.pth"))
    mvs_encoder = networks.FPN4(base_channels=8, scale=opt.prior_scale, dcn=opt.dcn)
    mvs_encoder.load_state_dict(mvs_encoder_dict, strict=True)
    mvs_encoder.eval()
    mvs_encoder.cuda()

    # reg 3d
    if opt.num_depth_bins >=8:
        reg3d = networks.reg3d(in_channels=opt.reg3d_c, base_channels=opt.reg3d_c, down_size=3)  # FIXME hardcoded
    else:
        reg3d = networks.reg2d(input_channel=opt.reg3d_c, base_channel=opt.reg3d_c)  # FIXME hardcoded
    reg3d_dict = torch.load(os.path.join(opt.load_weights_folder, "reg3d.pth"))
    reg3d.load_state_dict(reg3d_dict, strict=True)
    reg3d.eval()
    reg3d.cuda()
    
    # convex upsample
    if opt.convex_up:
        uplayer = convex_upsample_layer(feature_dim=8*2**opt.prior_scale, scale=opt.prior_scale)
        up_dict = torch.load(os.path.join(opt.load_weights_folder, "up.pth"))
        uplayer.load_state_dict(up_dict, strict=True)
        uplayer.eval()
        uplayer.cuda()

    mask_cnn = networks.UncertNet()
    mask_dict = torch.load(os.path.join(opt.load_weights_folder, "mask_cnn.pth"))
    mask_cnn.load_state_dict(mask_dict, strict=True)
    mask_cnn.eval()
    mask_cnn.cuda()

    pred_disps_z = []
    pred_disps_mono = []

    print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
    # do inference
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            input_color = data[('color', 0, 0)].cuda()
            output = mono_encoder(input_color)
            output = mono_depth_decoder(output)

            # predict poses
            pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
            if torch.cuda.is_available():
                pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
            for fi in frames_to_load[1:]:
                if fi < 0:
                    pose_inputs = [pose_feats[fi], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[fi]]

                pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                axisangle, translation = pose_dec(pose_inputs)
                pose = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=fi<0)
                data[('relative_pose', fi)] = pose
            relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
            relative_poses = torch.stack(relative_poses, 1)

            if torch.cuda.is_available():
                relative_poses = relative_poses.cuda()

            ref_match_feas, ref_context_feat = mvs_encoder(input_color)
            src_match_feats = []
            for f_i in opt.matching_ids[1:]:
                src_match_feat, _ = mvs_encoder(data["color_aug", f_i, 0].cuda())
                src_match_feats.append(src_match_feat)

            disp_prior = output[("disp", opt.prior_scale)].detach()
            disp_scaled = 1/opt.max_depth + disp_prior * (1/opt.min_depth - 1/opt.max_depth)
            depth_prior = 1 / disp_scaled

            ztrans = relative_poses[0,0,2,-1]  # B
            z_scale = opt.z_scale * ztrans
            mono_depth_prior_z = schedule_depth_range_zv2(depth_prior,
                                                    ndepth=opt.num_depth_bins, 
                                                    scale_fac=opt.depth_bin_fac, 
                                                    z_trans=z_scale)    # (B,D,H,W)
            
            cor_weight_sum_z = 1e-8
            cor_feats_z = 0
            for f_idx, _ in enumerate(opt.matching_ids[1:]):
                cost_vols_z = generate_costvol(ref_match_feas, src_match_feats[f_idx],
                                        data[('K', 2)].cuda(), data[('inv_K', 2)].cuda(),
                                        mono_depth_prior_z, relative_poses[:, f_idx:f_idx+1],
                                        opt.num_depth_bins,
                                        backprojector,
                                        projector)  # B D C H W
                B,D,C,H,W = cost_vols_z.shape
                cost_vols_z = cost_vols_z.reshape(B, D, -1, opt.reg3d_c, H, W).mean(2)  # B D G H W
                cor_weight_z = torch.softmax(cost_vols_z.mean(2), dim=1).max(1)[0]  # B H W
                cor_weight_sum_z += cor_weight_z  # B H W
                cor_feats_z += cor_weight_z.unsqueeze(1).unsqueeze(1) * cost_vols_z  # B D G H W


            # norm depth index
            cor_feats_z /= cor_weight_sum_z.unsqueeze(1).unsqueeze(1)
            cost_prob_z = reg3d(cor_feats_z)  # B D H W
            cost_prob_z = F.softmax(cost_prob_z, 1)
            depth_mvs_z = localmax(cost_prob_z, opt.norm_radius, opt.num_depth_bins, 1/mono_depth_prior_z[:,-1,:,:], 1/mono_depth_prior_z[:,0,:,:])  # B H W
        
            if opt.convex_up:
                depth_mvs_z = uplayer(depth_mvs_z, ref_context_feat)

            pred_disps_z.append(1/depth_mvs_z.cpu().numpy())
            pred_disp_mono, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp_mono = pred_disp_mono.cpu()[:, 0].numpy()
            pred_disps_mono.append(pred_disp_mono)

        pred_disps_z = np.concatenate(pred_disps_z)
        pred_disps_mono = np.concatenate(pred_disps_mono)


    gt_path = os.path.dirname(__file__) + "/splits/" + opt.eval_split + "/gt_depths.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    errors_z = []
    errors_mono = []
    errors_fuse = []
    for i in tqdm(range(pred_disps_mono.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp_z = np.squeeze(pred_disps_z[i])
        pred_disp_mono = np.squeeze(pred_disps_mono[i])
        pred_disp_mono = cv2.resize(pred_disp_mono, (gt_width, gt_height))
        pred_disp_z = cv2.resize(pred_disp_z, (gt_width, gt_height))
        pred_depth_z = 1 / pred_disp_z
        pred_depth_mono = 1 / pred_disp_mono

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth_z = pred_depth_z[mask]
        pred_depth_mono = pred_depth_mono[mask]
        gt_depth = gt_depth[mask]


        if not opt.disable_median_scaling:
            ratio1 = np.median(gt_depth) / np.median(pred_depth_mono)
            pred_depth_mono *= ratio1
            ratio2 = np.median(gt_depth) / np.median(pred_depth_z)
            pred_depth_z *= ratio2


        pred_depth_z[pred_depth_z < MIN_DEPTH] = MIN_DEPTH
        pred_depth_z[pred_depth_z > MAX_DEPTH] = MAX_DEPTH
        pred_depth_mono[pred_depth_mono < MIN_DEPTH] = MIN_DEPTH
        pred_depth_mono[pred_depth_mono > MAX_DEPTH] = MAX_DEPTH

        this_mvs_err_z = compute_errors(gt_depth, pred_depth_z)
        this_mono_err = compute_errors(gt_depth, pred_depth_mono)
        errors_fuse.append(compute_fuse_errors(gt_depth, pred_depth_mono, pred_depth_z))

        errors_z.append(this_mvs_err_z)
        errors_mono.append(this_mono_err)

    mean_errors_z = np.array(errors_z).mean(0)
    mean_errors_mono = np.array(errors_mono).mean(0)
    mean_errors_fuse = np.array(errors_fuse).mean(0)


    print('mono results:')
    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors_mono.tolist()) + "\\\\")
    print("\n")


    print('mvs results:')
    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors_z.tolist()) + "\\\\")
    print("\n")

    print('upbound results:')
    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors_fuse.tolist()) + "\\\\")
    print("\n")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
