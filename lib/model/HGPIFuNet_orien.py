import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
from .Embedder import *
from .. geometry import perspective_KRT

def positionalEncoder(cam_points, embedder, output_dim):
    cam_points = cam_points.permute(0, 2, 1)#[B,N,3]
    inputs_flat = torch.reshape(cam_points, [-1, cam_points.shape[-1]])
    embedded = embedder(inputs_flat)
    output = torch.reshape(embedded, [cam_points.shape[0], cam_points.shape[1], output_dim])
    return output.permute(0, 2, 1)

class HGPIFuNet_orien(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 #error_term=nn.MSELoss(),
                 error_term=nn.L1Loss(),
                 ):
        super(HGPIFuNet_orien, self).__init__(
            error_term=error_term)

        self.name = 'hgpifu'
        

        self.opt = opt
        self.gen_orien = self.opt.gen_orien

        self.image_filter = HGFilter(opt)

        self.orign_embedder = None
        self.embedder_outDim = None
        mlp_dim = self.opt.mlp_dim_orien

        # if self.opt.positional_encoding:
        self.orign_embedder, self.embedder_outDim = get_embedder(3)
        self.embedder = positionalEncoder
        mlp_dim[0] = mlp_dim[0] + self.embedder_outDim

        if self.gen_orien:
            mlp_dim[-1] = 3

        self.surface_classifier = SurfaceClassifier(
            filter_channels=mlp_dim,
            no_residual=self.opt.no_residual,
            last_op=None)#nn.Sigmoid() for occ

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        # xyz = self.projection(points, calibs, transforms)
        # xyz = orthogonal(points, calibs, transforms)
        # xyz = perspective_rev(points, calibs, self.opt.img_width, self.opt.img_height, self.opt.crop_bbox[0], self.opt.crop_bbox[1], self.opt.crop_bbox[2], self.opt.crop_bbox[3])
        # xyz = orthogonal_rev(points, calibs, self.opt.img_width, self.opt.img_height, self.opt.crop_bbox[0], self.opt.crop_bbox[1], self.opt.crop_bbox[2], self.opt.crop_bbox[3])
        # xyz = perspective_nocrop(points, calibs)  
        xyz = perspective_KRT(self.opt.img_mode, points, calibs, self.opt.is_wild)   # For synthetic EBStore data 

        # print(xyz.shape)
        xy = xyz[:, :2, :]#/2.0
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        position_feat = self.embedder(points, self.orign_embedder, self.embedder_outDim)
        #position_feat = self.embedder(xyz, self.orign_embedder, self.embedder_outDim)

        # z_feat = self.normalizer(z)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []
        #print(len(self.im_feat_list))
        for im_feat in self.im_feat_list:
            # [B, Feat_i + z, N]
            point_local_feat_list = [self.index(im_feat, xy), position_feat]#, z_feat #mlp_dim 257->256
            # point_local_feat_list = [self.index(im_feat, xy), z_feat]
            #point_local_feat_list = [points, position_feat]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            #pred = self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

        return self.preds

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)
        
        return error

    def forward(self, images, points, calibs, labels=None):
        # Get image feature
        self.filter(images)
        #self.im_feat_list = [0]

        # Phase 2: point query
        self.query(points=points, calibs=calibs, labels=labels)

        # get the prediction
        res = self.get_preds()
        
        # get the error
        error = self.get_error()

        return res, error