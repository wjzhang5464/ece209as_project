import torch
import torch.nn as nn
import torch.nn.functional as F
from select_backbone import select_resnet
import math

class Multimodal(nn.Module):
	def __init__(self, img_dim, network='resnet50', num_layers_in_fc_layers = 1024, dropout=0.5, latent_dim= 1024, lstm_layers=1 , hidden_dim = 1024, bidirectional = False):
		super(Multimodal, self).__init__();

		self.__nFeatures__ = 24;
		self.__nChs__ = 32;
		self.__midChs__ = 32;

		self.netcnnaud = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

			nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
			nn.BatchNorm2d(192),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

			nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

			nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
			nn.BatchNorm2d(512),
			nn.ReLU(),
		);

		self.netfcaud = nn.Sequential(
			nn.Linear(512*21, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, num_layers_in_fc_layers),
		);

		self.netcnnlip, self.param = select_resnet(network, track_running_stats=False);
		self.last_duration = int(math.ceil(30 / 4))
		self.last_size = int(math.ceil(img_dim / 32))

		self.netfclip = nn.Sequential(
			nn.Linear(self.param['feature_size']*self.last_size*self.last_size, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, num_layers_in_fc_layers),
			nn.BatchNorm1d(num_layers_in_fc_layers),
			nn.ReLU(),
		);

		self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
		self.final_bn_lip.weight.data.fill_(1)
		self.final_bn_lip.bias.data.zero_()

		self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
		self._initialize_weights(self.final_fc_lip)

		self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
		self.final_bn_aud.weight.data.fill_(1)
		self.final_bn_aud.bias.data.zero_()

		self.final_fc_aud = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
		self._initialize_weights(self.final_fc_aud)


		self._initialize_weights(self.netcnnaud)
		self._initialize_weights(self.netfcaud)
		self._initialize_weights(self.netfclip)


		self.dp = nn.Dropout(0.4)
		self.linear1 = nn.Linear(1024, 2)#2 is number of classes
		self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
		

	def forward_aud(self, x):
		(B, N, C, H, W) = x.shape
		x = x.view(B*N, C, H, W)
		mid = self.netcnnaud(x);
		mid = mid.view((mid.size()[0], -1));
		x = self.netfcaud(mid);
		x = x.view(B, N, 1024)
		x_lstm, _ = self.lstm(x, None)
		return torch.mean(x_lstm, dim=1);

	def forward_vid(self, x):
		(B, N, C, NF, H, W) = x.shape
		x = x.view(B*N, C, NF, H, W)
		feature = self.netcnnlip(x);
		feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
		feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size)
		feature = feature.view((feature.size()[0]*N, -1));
		x = self.netfclip(feature);
		x = x.view(B, N, 1024)
		x_lstm, _ = self.lstm(x, None)
		return torch.mean(x_lstm,dim = 1);



	def final_classification_vid(self,feature):
		output = self.dp(self.linear1(feature))
		return output

	def final_classification_aud(self,feature):
		output = self.dp(self.linear1(feature))
		return output


	def _initialize_weights(self, module):
		for m in module:
			if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.ReLU) or isinstance(m,nn.MaxPool2d) or isinstance(m,nn.Dropout):
				pass
			else:
				m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None: m.bias.data.zero_()