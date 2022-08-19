import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

import json

from .utils import readlines, sec_to_hm_str, seed_worker, get_dist_info, DistributedSampler
from .layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors, schedule_depth_range_zv2, schedule_depth_rangev2, convex_upsample_layer, \
    generate_costvol, localmax, reproject_with_depth, random_image_mask, entropy

from movedepth import datasets, networks
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.mvs_parameters_to_train = []

        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        if self.opt.ddp:
            dist.init_process_group(backend='nccl', )
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = self.opt.matching_ids
        if self.local_rank == 0:
            print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        # mono encoder
        self.models["mono_encoder"] = networks.ResnetEncoder(self.opt.res_arch, self.opt.weights_init == "pretrained")
        if self.opt.ddp:
            self.models["mono_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["mono_encoder"])
        self.models["mono_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["mono_encoder"].parameters())
        
        # mono depth decoder
        self.models["mono_depth"] = networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales, 
                    match_conv=False, ddv=False, discret=None, mono_conf=False, mono_bins=False)
        if self.opt.ddp:
            self.models["mono_depth"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["mono_depth"])
        self.models["mono_depth"].to(self.device)
        self.parameters_to_train += list(self.models["mono_depth"].parameters())

        # load pose is worse than joint training!
        if not self.opt.load_pose:
            # pose encoder
            self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.res_arch, self.opt.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            if self.opt.ddp:
                self.models["pose_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose_encoder"])
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            
            # pose decoder
            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
            if self.opt.ddp:
                self.models["pose"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose"])
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            # fuse mask
        self.models["mask_cnn"] = networks.UncertNet()
        if self.opt.ddp:
            self.models["mask_cnn"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["mask_cnn"])
        self.models["mask_cnn"].to(self.device)
        self.mvs_parameters_to_train += list(self.models["mask_cnn"].parameters())

        # mvs decoder
        self.models["mvs_encoder"] = networks.FPN4(base_channels=8, scale=self.opt.prior_scale, dcn=self.opt.dcn)
        if self.opt.ddp:
            self.models["mvs_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["mvs_encoder"])
        self.models["mvs_encoder"].to(self.device)
        self.mvs_parameters_to_train += list(self.models["mvs_encoder"].parameters())

        # mvs conv3d
        self.backprojector = BackprojectDepth(batch_size=self.opt.num_depth_bins, height=self.opt.height//2**(self.opt.prior_scale), width=self.opt.width//2**(self.opt.prior_scale))
        self.projector = Project3D(batch_size=self.opt.num_depth_bins, height=self.opt.height//2**(self.opt.prior_scale), width=self.opt.width//2**(self.opt.prior_scale))
        self.backprojector.to(self.device)
        self.projector.to(self.device)
        if self.opt.num_depth_bins >=8:
            self.models["reg3d"] = networks.reg3d(in_channels=self.opt.reg3d_c, base_channels=self.opt.reg3d_c, down_size=3)  # FIXME hardcoded
        else:
            self.models["reg3d"] = networks.reg2d(input_channel=self.opt.reg3d_c, base_channel=self.opt.reg3d_c)  # FIXME hardcoded
        if self.opt.ddp:
            self.models["reg3d"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["reg3d"])
        self.models["reg3d"].to(self.device)
        self.mvs_parameters_to_train += list(self.models["reg3d"].parameters())

        # upsample
        if self.opt.convex_up:
            self.models["up"] = convex_upsample_layer(feature_dim=8*2**self.opt.prior_scale, scale=self.opt.prior_scale)
            if self.opt.ddp:
                self.models["up"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["up"])
            self.models["up"].to(self.device)
            self.parameters_to_train += list(self.models["up"].parameters())

        if self.opt.ddp:
            for key in self.models.keys():
                self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        
        self.model_optimizer = optim.Adam([
            {'params': self.parameters_to_train, 'lr': self.opt.learning_rate},
            {'params': self.mvs_parameters_to_train, 'lr': self.opt.learning_rate*self.opt.lr_fac},
        ])
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        if self.local_rank == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
            print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        if self.opt.dataset.startswith('kitti'):
            fpath = os.path.dirname(__file__) + "/splits/{}"+"/{}_files.txt"
            train_filenames = readlines(fpath.format(self.opt.split, "train"))
            val_filenames = readlines(fpath.format(self.opt.split, "val"))
        else:
            assert False, "Not implemented yet"
        img_ext = '.png' if self.opt.png else '.jpg'

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext, load_pose=self.opt.load_pose)

        if self.opt.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker)

        self.num_total_steps = len(self.train_loader) * self.opt.num_epochs

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext, load_pose=self.opt.load_pose)

        if self.opt.ddp:
            rank, world_size = get_dist_info()
            self.world_size = world_size
            val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False, sampler=val_sampler)
        else:
            self.world_size = 1
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.val_iter = iter(self.val_loader)
        self.writers = {}
        for mode in ["train", "val"]:
            if self.local_rank == 0:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if self.local_rank == 0:
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))
        if self.opt.ddp:
            self.opt.log_frequency = self.opt.log_frequency//self.world_size
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for k, m in self.models.items():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.opt.ddp:
                self.train_loader.sampler.set_epoch(self.epoch)

            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.epoch>15:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        if self.local_rank == 0:
            print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                if self.local_rank == 0:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                if self.local_rank == 0:
                    self.log("train", inputs, outputs, losses)
                self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            self.step += 1
        self.model_lr_scheduler.step()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        if not self.opt.load_pose:
            pose_pred = self.predict_poses(inputs, None)
            outputs.update(pose_pred)

        else:
            for f_i in self.opt.frame_ids[1:]:
                outputs[("cam_T_cam", 0, f_i)] = inputs['relative_pose', f_i]

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        # mvs feature extraction
        ref_match_feat, ref_context_feat = self.models["mvs_encoder"](inputs["color_aug", 0, 0])
        src_match_feats = []
        for f_i in self.matching_ids[1:]:
            src_match_feat, _ = self.models["mvs_encoder"](inputs["color_aug", f_i, 0])
            src_match_feats.append(src_match_feat)

        # single frame path
        feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
        outputs.update(self.models['mono_depth'](feats, no_match=False))

        # mono reprojection loss
        self.generate_images_pred(inputs, outputs)
        mono_losses = self.compute_losses(inputs, outputs)

        # the mono depth prior for mvs
        disp_prior = outputs[("disp", self.opt.prior_scale)].clone().detach()
        disp_scaled = 1/self.opt.max_depth + disp_prior * (1/self.opt.min_depth - 1/self.opt.max_depth)
        depth_prior = 1 / disp_scaled
        if self.epoch > self.opt.ztrans_start_epc:
            mono_depth_prior = schedule_depth_range_zv2(depth_prior,
                                                    ndepth=self.opt.num_depth_bins, 
                                                    scale_fac=self.opt.depth_bin_fac,
                                                    z_trans=self.opt.z_scale*relative_poses[:,:,2:3,-1:],
                                                    type=self.opt.schedule_type)    # (B,D,H,W)
        else:
            mono_depth_prior = schedule_depth_rangev2(depth_prior,
                                        ndepth=self.opt.num_depth_bins, 
                                        scale_fac=self.opt.depth_bin_fac,
                                        type=self.opt.schedule_type)    # (B,D,H,W)

        # mvs cost volume
        cor_weight_sum = 1e-8
        cor_feats = 0
        for f_idx, _ in enumerate(self.matching_ids[1:]):
            cost_vols = generate_costvol(ref_match_feat, src_match_feats[f_idx],
                                    inputs[('K', 2)], inputs[('inv_K', 2)],
                                    mono_depth_prior, relative_poses[:, f_idx:f_idx + 1],
                                    self.opt.num_depth_bins,
                                    self.backprojector,
                                    self.projector)  # B D C H W
            B, D, C, H, W = cost_vols.shape
            cost_vols = cost_vols.reshape(B, D, -1, self.opt.reg3d_c, H, W).mean(2)  # B D G H W
            cor_weight = torch.softmax(cost_vols.mean(1), dim=1).max(1)[0]  # B H W
            cor_weight_sum += cor_weight  # B H W
            cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cost_vols  # B D G H W
        cor_feats /= cor_weight_sum.unsqueeze(1).unsqueeze(1)

        # norm depth index
        cost_prob = self.models["reg3d"](cor_feats)  # B D H W
        cost_prob = F.softmax(cost_prob, 1)
        # mvs fuse mask
        cost_prob_entropy = entropy(cost_prob, dim=1, keepdim=True)  # B 1 H W
        trust_mono_mask = self.models["mask_cnn"](cost_prob_entropy)  # B 1 H W
        depth_mvs = localmax(cost_prob, self.opt.norm_radius, self.opt.num_depth_bins, 1/mono_depth_prior[:,-1,:,:], 1/mono_depth_prior[:,0,:,:])  # B H W

        # mask aug depth prediction
        ori_H, ori_W = inputs["color_aug", 0, 0].shape[2:]
        masked_img, this_aug_mask = random_image_mask(inputs["color_aug", 0, 0], [ori_H//3, ori_W//3])
        ref_aug_feat, _ = self.models["mvs_encoder"](masked_img)
        cor_weight_sum = 1e-8
        cor_feats = 0
        for f_idx, _ in enumerate(self.matching_ids[1:]):
            cost_vols = generate_costvol(ref_aug_feat, src_match_feats[f_idx],
                                    inputs[('K', 2)], inputs[('inv_K', 2)],
                                    mono_depth_prior, relative_poses[:, f_idx:f_idx + 1],
                                    self.opt.num_depth_bins,
                                    self.backprojector,
                                    self.projector)  # B D C H W
            B, D, C, H, W = cost_vols.shape
            cost_vols = cost_vols.reshape(B, D, -1, self.opt.reg3d_c, H, W).mean(2)  # B D G H W
            cor_weight = torch.softmax(cost_vols.mean(1), dim=1).max(1)[0]  # B H W
            cor_weight_sum += cor_weight  # B H W
            cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cost_vols  # B D G H W
        cor_feats /= cor_weight_sum.unsqueeze(1).unsqueeze(1)

        # norm depth index
        cost_prob = self.models["reg3d"](cor_feats)  # B D H W
        cost_prob = F.softmax(cost_prob, 1)
        depth_mvs_aug = localmax(cost_prob, self.opt.norm_radius, self.opt.num_depth_bins, 1/mono_depth_prior[:,-1,:,:], 1/mono_depth_prior[:,0,:,:])  # B H W

        this_mask = F.interpolate(this_aug_mask, [depth_mvs_aug.shape[1], depth_mvs_aug.shape[2]], mode="bilinear", align_corners=True).sum(1).to(torch.bool)
        this_masked_loss = F.smooth_l1_loss(depth_mvs_aug[this_mask], depth_mvs[this_mask], size_average=True) * self.opt.mask_lw
        mono_losses["masked_loss"] = this_masked_loss * self.opt.mask_lw
        mono_losses["loss"] += mono_losses["masked_loss"]
        outputs["masked_depth"] = depth_mvs_aug
        outputs["masked_aug"] = this_aug_mask

        # upsample mvs depth
        if not self.opt.convex_up:
            depth_mvs = F.interpolate(depth_mvs.unsqueeze(1), [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)[:, 0]
        else:
            depth_mvs = self.models["up"](depth_mvs, ref_context_feat)
        outputs['depth_mvs'] = depth_mvs
        _, mono_depth = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        trust_mono_mask = F.interpolate(trust_mono_mask, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
        fused_depth = (1-trust_mono_mask) * depth_mvs[:,None].detach() + trust_mono_mask * mono_depth.detach()
        outputs["fused_depth"] = fused_depth
        outputs["trust_mono_mask"] = trust_mono_mask
        fuse_losses = self.compute_fuse_losses(inputs, outputs)

        # mvs mask
        if self.opt.mask_mvs_conf:
            cost_prob = F.interpolate(cost_prob.unsqueeze(1), [D, self.opt.height, self.opt.width], mode="trilinear", align_corners=True)  # B 1 D H W
            photo_conf_map = cost_prob.max(2)[0] > self.opt.photo_conf  # B 1 H W
            outputs['photo_conf_map'] = photo_conf_map  # B 1 H W
        if self.opt.mask_mvs_dist:
            dist_mask = outputs[("disp", 0)] > self.opt.dist_thres
            outputs["dist_mask"] = dist_mask  # B 1 H W

        self.generate_images_pred(inputs, outputs, is_mvs=True)
        mvs_losses = self.compute_losses(inputs, outputs, is_mvs=True)
        for key, val in mono_losses.items():
            if key in mvs_losses.keys():
                mvs_losses[key] += val
            else:
                mvs_losses[key] = val


        for key, val in fuse_losses.items():
            if key in mvs_losses.keys():
                mvs_losses[key] += val
            else:
                mvs_losses[key] = val

        return outputs, mvs_losses


    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        for fi in self.matching_ids[1:]:
            inputs[('relative_pose', fi)] = outputs[("cam_T_cam", 0, fi)].clone().detach()

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            if self.local_rank == 0:
                self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, is_mvs=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        if is_mvs:
            depth_mvs = outputs["depth_mvs"]
            for _, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                T = T.detach()
                source_scale = 0
                cam_points = self.backproject_depth[source_scale](depth_mvs, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                outputs[("mvs_mask", frame_id)] = ((pix_coords<-1) | (pix_coords>1)).sum(-1) > 0  # 1 H W
                outputs[("mvs_color", frame_id)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border", align_corners=True)
            return

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]


    def compute_reprojection_loss(self, pred, target, ssim_lw=None):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            if ssim_lw is None:
                reprojection_loss = self.opt.ssim_lw * ssim_loss + (1-self.opt.ssim_lw) * l1_loss
            else:
                reprojection_loss = ssim_lw * ssim_loss + (1-ssim_lw) * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_fuse_losses(self, inputs, outputs):
        depth_fuse = outputs["fused_depth"]
        # for _, frame_id in enumerate(self.matching_ids[1:]):
        for frame_id in self.opt.frame_ids[1:]:
            T = outputs[("cam_T_cam", 0, frame_id)].detach()
            source_scale = 0
            cam_points = self.backproject_depth[source_scale](depth_fuse, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
            outputs[("mvs_color_fuse", frame_id)] = F.grid_sample(
                inputs[("color", frame_id, source_scale)],
                pix_coords,
                padding_mode="border", align_corners=True)

        losses = {}
        reprojection_losses = []
        scale = 0
        target = inputs[("color", 0, source_scale)]
        for frame_id in self.opt.frame_ids[1:]:
            pred = outputs[("mvs_color_fuse", frame_id)]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target, ssim_lw=0))
        reprojection_losses = torch.cat(reprojection_losses, 1)  # B N H W

        # auto mask
        if self.opt.mask_mvs_auto:
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target, ssim_lw=0))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss, identity_reprojection_loss)
        else:
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)  # B 1 H W
            reprojection_loss_mask = torch.ones_like(reprojection_loss)
        outputs["reprojection_loss_mask"] = reprojection_loss_mask

        reprojection_loss = reprojection_loss * reprojection_loss_mask
        reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)
        losses['fuse_reproj_loss'] = reprojection_loss
        loss = reprojection_loss
        losses["loss"] = loss
        return losses

    def compute_losses(self, inputs, outputs, is_mvs=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        reprojection_losses = []

        if is_mvs:
            scale = 0
            loss = 0
            source_scale = 0
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("mvs_color", frame_id)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # B N H W

            # auto mask
            if self.opt.mask_mvs_auto:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
                reprojection_loss_mask = self.compute_loss_masks(reprojection_loss, identity_reprojection_loss)
            else:
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)  # B 1 H W
            outputs["mvs_reprojection_loss"] = reprojection_loss

            reprojection_loss_mask = torch.ones_like(reprojection_loss)

            # mvs mask loss
            if self.opt.mask_mvs_conf:
                reprojection_loss_mask = reprojection_loss_mask * outputs["photo_conf_map"]
            if self.opt.mask_mvs_dist:
                reprojection_loss_mask = reprojection_loss_mask * outputs["dist_mask"]
            if self.opt.mask_mvs_geo:
                for f_id in self.opt.frame_ids[1:]:
                    reprojection_loss_mask = reprojection_loss_mask * outputs[("geo_mask", f_id)]
            outputs["reprojection_loss_mask"] = reprojection_loss_mask

            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)
            outputs['mvs_reproj_loss'] = reprojection_loss
            loss += reprojection_loss

            # smooth loss
            if self.opt.mvs_smooth_loss:
                depth_mvs = outputs["depth_mvs"].unsqueeze(1)
                mean_depth = depth_mvs.mean(2, True).mean(3, True)
                norm_depth = depth_mvs / (mean_depth + 1e-7)
                smooth_loss = get_smooth_loss(norm_depth, color)
                losses['mvs_smooth_loss/{}'.format(scale)] = smooth_loss
                loss += self.opt.disparity_smoothness * smooth_loss
            losses["loss"] = loss
            return losses

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # B 2 H W

            # auto mask
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
                reprojection_loss_mask = self.compute_loss_masks(reprojection_loss, identity_reprojection_loss)
            else:
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
                reprojection_loss_mask = torch.ones_like(reprojection_loss)

            if scale == 0:
                outputs['mono_reproj_loss'] = reprojection_loss

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            loss += reprojection_loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            losses['mono_smooth_loss/{}'.format(scale)] = smooth_loss

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        if self.local_rank == 0:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                writer.add_image("color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                if s == 0 and frame_id != 0:
                    writer.add_image("color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)

            disp = colormap(outputs[('disp', s)][j, 0])
            writer.add_image( "disp_mono/{}".format(j), disp, self.step)

            disp_mvs = 1 / outputs["depth_mvs"]
            disp_mvs = colormap(disp_mvs[j])
            writer.add_image( "disp_mvs/{}".format(j), disp_mvs, self.step)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if self.local_rank == 0:
            if save_step:
                save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch, self.step))
            else:
                save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
            
            if self.epoch == self.opt.num_epochs-1:
                save_folder = os.path.join(self.log_path, "models", "last")

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                if self.opt.ddp:
                    to_save = model.module.state_dict()
                else:
                    to_save = model.state_dict()
                torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)

    def load_mono_model(self):
        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            if self.local_rank == 0:
                print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        if self.local_rank == 0:
            print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            if self.local_rank == 0:
                print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            if self.opt.ddp:
                pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
            else:
                pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
