import torch
import torch.nn as nn
from ..net_util import init_net
from .LengthClassifier import LengthClassifier
from .HGFilters import *
from .Embedder import *
from ..geometry import perspective_KRT, index
from .EncoderRNN import EncoderRNN
import random


def positionalEncoder(cam_points, embedder, output_dim):
    cam_points = cam_points.permute(0, 2, 1) # [B, N, 3]
    inputs_flat = torch.reshape(cam_points, [-1, cam_points.shape[-1]])
    embedded = embedder(inputs_flat)
    output = torch.reshape(embedded, [cam_points.shape[0], cam_points.shape[1], output_dim])
    return output.permute(0, 2, 1)

class Len_Cls_Net(nn.Module):

    def __init__(self,
                 opt,
                # error_term=nn.CrossEntropyLoss(),
                error_term=nn.BCEWithLogitsLoss()
                 ):
        super(Len_Cls_Net, self).__init__()

        self.name = 'len_class_net'
        self.opt = opt
        self.error_term = error_term

        self.image_filter = HGFilter(opt)
        
        self.orign_embedder = None
        self.embedder_outDim = None

        self.orign_embedder, self.embedder_outDim = get_embedder(3)
        self.embedder = positionalEncoder

        mlp_dim = self.opt.mlp_dim_length

        if opt.not_use_PEDE:
            self.rnn_encoder = EncoderRNN(mlp_dim[0])
        elif opt.not_use_DE:
            self.rnn_encoder = EncoderRNN(mlp_dim[0]+self.embedder_outDim)
        else:
            self.rnn_encoder = EncoderRNN(mlp_dim[0]+2*self.embedder_outDim)

        self.length_classifier = LengthClassifier(
            filter_channels=mlp_dim,
            no_residual=self.opt.no_residual,
            last_op=None) # CE loss inherently includes Softmax, while BCE with Logits inherently includes Sigmoid.
        
        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        init_net(self)
    
    def img_filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # self.im_feat_list, self.tmpx, self.normx = self.image_encoder(images)
        # if not self.training:
        self.im_feat_list = [self.im_feat_list[-1]]
            
    
    def get_token(self, length, is_train=True, max_len=13):
        # Length: Nx1 token: 2Nx2 label: 2Nx1
        token_list = []
        label_list = []

        if is_train:
            for k in range(length.shape[1]):  # length: [1, N]
                # Make positive sample
                i = random.randint(1, int(length[0][k])-1)
                token_list.append([k, i])
                label_list.append(1)

                # Make negative sample
                token_list.append([k, length[0][k]])
                label_list.append(0)
        
        else:
            for k in range(length.shape[1]):
                for i in range(1, int(length[0][k])):
                    token_list.append([k, i])
                    label_list.append(1)
                for i in range(int(length[0][k]), max_len+1):
                    token_list.append([k, i])
                    label_list.append(0)

        tokens = torch.Tensor(token_list).int()
        labels = torch.Tensor(label_list).unsqueeze(1).float()
        return tokens, labels

    def get_token_pseudo(self, length, is_train=True, max_len=13):
        # Length: Nx1 token: 2Nx2 label: 2Nx1
        token_list = []
        label_list = []

        if is_train:
            for k in range(length.shape[1]):
                # Make positive sample
                i = random.randint(1, int(length[0][k])-1)
                token_list.append([k, i])
                label_list.append(1)

                # Make negative sample
                i = random.randint(int(length[0][k]), max_len)
                token_list.append([k, i])
                label_list.append(0)
        
        else:
            for k in range(length.shape[1]):
                for i in range(1, int(length[0][k])):
                    token_list.append([k, i])
                    label_list.append(1)
                token_list.append([k, length[0][k]])
                label_list.append(0)

        tokens = torch.Tensor(token_list).int()
        labels = torch.Tensor(label_list).unsqueeze(1).float()
        return tokens, labels


    def infer_rnn_encoder(self, points, calibs, token, num_per_strand):
        xyz = perspective_KRT(self.opt.img_mode, points, calibs)
        xy = xyz[:, :2, :]
        k = token[0]
        m = token[1]
        hidden_feat_list = []

        for im_feat in self.im_feat_list:
            strand_im_feat = index(im_feat, xy)
            strand_im_feat = torch.reshape(strand_im_feat, (strand_im_feat.shape[0], strand_im_feat.shape[1], int(strand_im_feat.shape[2] / num_per_strand), num_per_strand))

            if self.opt.not_use_PEDE:
                point_local_feat_list = [strand_im_feat[:, :, k, :m]]
            elif self.opt.not_use_DE:
                point_local_feat_list = [strand_im_feat[:, :, k, :m], self.position_feat[:, :, k, :m]]
            else:
                point_local_feat_list = [strand_im_feat[:, :, k, :m], self.position_feat[:, :, k, :m], self.direction_feat[:, :, k, :m]]
            point_local_feat = torch.cat(point_local_feat_list, 1).permute(2, 0, 1) # N-parts, 1, C

            encoder_init_hidden = self.rnn_encoder.init_hidden.repeat(1, 1, 1).to(torch.device('cuda:%d' % point_local_feat.get_device()))
            _, hidden = self.rnn_encoder(point_local_feat, encoder_init_hidden)
            hidden_feat_list.append(hidden)

        return hidden_feat_list


    def get_preds(self, points, calibs, tokens, labels, num_per_strand, thresh):
        self.eyeb_error_list = []
        eyeb_hidden_list = []

        for idx in range(tokens.shape[0]):
            token = tokens[idx]
            hidden_feat_list = self.infer_rnn_encoder(points, calibs, token, num_per_strand)
            hidden = hidden_feat_list[-1]
            eyeb_hidden_list.append(hidden)

        eyeb_hidden = torch.cat(eyeb_hidden_list, dim=1).permute(0, 2, 1)
        preds = self.length_classifier(eyeb_hidden).squeeze()
        labels = labels.to(torch.device('cuda:%d' % preds.get_device())).squeeze()
        eyeb_error = self.error_term(preds, labels)

        sigmoid = nn.Sigmoid()
        pred_labels = (sigmoid(preds) >= thresh).float()

        eyeb_acc = torch.sum(pred_labels==labels) / pred_labels.shape[0]
        eyeb_positive_acc = torch.sum((pred_labels==labels) & (labels==1))/torch.sum((labels==1))
        eyeb_negative_acc = torch.sum((pred_labels==labels) & (labels==0))/torch.sum((labels==0))

        eyeb_token_count = pred_labels.shape[0]
        eyeb_positive_count = torch.sum((labels==1))
        eyeb_negative_count = torch.sum((labels==0))

        return pred_labels, eyeb_error, eyeb_acc, eyeb_negative_acc, eyeb_positive_acc, eyeb_token_count, eyeb_positive_count, eyeb_negative_count

    
    def forward(self, images, points, orien, calibs, length, is_train=True):
        # Get image feature list
        self.img_filter(images)

        # Get position feature
        num_per_strand = points.shape[2]
        points = torch.reshape(points, (points.shape[0], points.shape[1]*points.shape[2], 3)).permute(0, 2, 1)
        self.position_feat = self.embedder(points, self.orign_embedder, self.embedder_outDim)
        self.position_feat = torch.reshape(self.position_feat, (self.position_feat.shape[0], self.position_feat.shape[1], int(self.position_feat.shape[2] / num_per_strand), num_per_strand))

        # Get direction feature
        orien = torch.reshape(orien, (orien.shape[0], orien.shape[1]*orien.shape[2], 3)).permute(0, 2, 1)
        self.direction_feat = self.embedder(orien, self.orign_embedder, self.embedder_outDim)
        self.direction_feat = torch.reshape(self.direction_feat, (self.direction_feat.shape[0], self.direction_feat.shape[1], int(self.direction_feat.shape[2] / num_per_strand), num_per_strand))

        # Get token
        tokens, gt_labels = self.get_token(length, is_train)
    
        eyeb_pred_labels, eyeb_error, eyeb_acc, eyeb_negative_acc, eyeb_positive_acc, eyeb_token_count, eyeb_positive_count, eyeb_negative_count = self.get_preds(points, calibs, tokens, gt_labels, num_per_strand, self.opt.len_class_thres)

        return eyeb_pred_labels, eyeb_error, eyeb_acc, eyeb_negative_acc, eyeb_positive_acc, eyeb_token_count, eyeb_positive_count, eyeb_negative_count