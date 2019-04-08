
import numpy as np 
import cv2
import torch
import sys
import os
import time

import fhog
from ..net.models import SiamFC_Incep22, SiamFC_Res22
from ... import Tracker


# ffttools
def fftd(img, backwards=False):	
	# shape of img can be (m,n), (m,n,1) or (m,n,2)	
	# in my test, fft provided by numpy and scipy are slower than cv2.dft
	return cv2.dft(np.float32(img), flags = ((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))   # 'flags =' is necessary!
	
def real(img):
	return img[:,:,0]
	
def imag(img):
	return img[:,:,1]
		
def complexMultiplication(a, b):
	res = np.zeros(a.shape, a.dtype)
	
	res[:,:,0] = a[:,:,0]*b[:,:,0] - a[:,:,1]*b[:,:,1]
	res[:,:,1] = a[:,:,0]*b[:,:,1] + a[:,:,1]*b[:,:,0]
	return res

def complexDivision(a, b):
	res = np.zeros(a.shape, a.dtype)
	divisor = 1. / (b[:,:,0]**2 + b[:,:,1]**2)
	
	res[:,:,0] = (a[:,:,0]*b[:,:,0] + a[:,:,1]*b[:,:,1]) * divisor
	res[:,:,1] = (a[:,:,1]*b[:,:,0] + a[:,:,0]*b[:,:,1]) * divisor
	return res

def rearrange(img):
	#return np.fft.fftshift(img, axes=(0,1))
	assert(img.ndim==2), 'input is of shape {}, not 2 channel'.format(img.shape)
	img_ = np.zeros(img.shape, img.dtype)
	xh, yh = img.shape[1]/2, img.shape[0]/2
	img_[0:yh,0:xh], img_[yh:img.shape[0],xh:img.shape[1]] = img[yh:img.shape[0],xh:img.shape[1]], img[0:yh,0:xh]
	img_[0:yh,xh:img.shape[1]], img_[yh:img.shape[0],0:xh] = img[yh:img.shape[0],0:xh], img[0:yh,xh:img.shape[1]]
	return img_


# recttools
def x2(rect):
	return rect[0] + rect[2]

def y2(rect):
	return rect[1] + rect[3]

def limit(rect, limit):
	if(rect[0]+rect[2] > limit[0]+limit[2]):
		rect[2] = limit[0]+limit[2]-rect[0]
	if(rect[1]+rect[3] > limit[1]+limit[3]):
		rect[3] = limit[1]+limit[3]-rect[1]
	if(rect[0] < limit[0]):
		rect[2] -= (limit[0]-rect[0])
		rect[0] = limit[0]
	if(rect[1] < limit[1]):
		rect[3] -= (limit[1]-rect[1])
		rect[1] = limit[1]
	if(rect[2] < 0):
		rect[2] = 0
	if(rect[3] < 0):
		rect[3] = 0
	return rect

def getBorder(original, limited):
	res = [0,0,0,0]
	res[0] = limited[0] - original[0]
	res[1] = limited[1] - original[1]
	res[2] = x2(original) - x2(limited)
	res[3] = y2(original) - y2(limited)
	assert(np.all(np.array(res) >= 0))
	return res

def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
	cutWindow = [x for x in window]
	limit(cutWindow, [0,0,img.shape[1],img.shape[0]])   # modify cutWindow
	assert(cutWindow[2]>0 and cutWindow[3]>0)
	border = getBorder(window, cutWindow)
	res = img[cutWindow[1]:cutWindow[1]+cutWindow[3], cutWindow[0]:cutWindow[0]+cutWindow[2]]

	if(border != [0,0,0,0]):
		res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
	return res

## for loading pretrained model
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(len(missing_keys)))
    #print(missing_keys)
    print('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



# KCF tracker
class KCFTracker(Tracker):
	def __init__(self):
		super(KCFTracker, self).__init__(name="DeepKCF", is_deterministic = True)

		self.lambdar = 0.0001   # regularization
		self.padding = 2.5   # extra area surrounding the target
		self.restrict_height = 1.5
		self.restrict_width = 3.0
		self.restrict_large = 2.0
		self.output_sigma_factor = 0.125   # bandwidth of gaussian target

		self.interp_factor = 0.01 #0.012   # linear interpolation factor for adaptation
		#self.sigma = 0.6  # gaussian kernel bandwidth
		# TPAMI   #interp_factor = 0.02   #sigma = 0.5
		self.cell_size = 4   # HOG cell size

		#net
		self.net = SiamFC_Res22()
		self.pretrain_path = './got10k/trackers/DeepKCF/pretrain/CIResNet22.pth'
		self.layer_size = [ 64, 256, 512 ]
		self.indLayers = [19, 28, 37]     #The CNN layers Conv5-4, Conv4-4, and Conv3-4 in ResNet
		self.nweights  = [0.25, 0.5, 1]   #Weights for combining correlation filter responses
		self.numLayers = len(self.indLayers)
		self.init_net()

		# multiscale
		self.net_insize = 255
		self.template_size = 1   # template size
		self.scale_step = [1.01, 1.02, 1.03, 1.04] #1.05   # scale step for multi-scale estimation
		self.scale_weight = [0.999, 0.996, 0.993, 0.990] #1 # to downweight detection scores of other scales for added stability

		#tracking params
		self.hann = None
		self._tmpl_sz = [0,0]  # cv::Size, [width,height]  #[int,int]
		self._roi = [0.,0.,0.,0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
		self._scale = 1.   # float
		self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
		self._xf = None
		self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
		#self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
		#self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])


	def init_net(self):
		assert os.path.isfile(self.pretrain_path), '{} is not a valid file'.format(self.pretrain_path)
		self.net = load_pretrain(self.net, self.pretrain_path)
		self.net.eval()
		self.net = self.net.cuda()


	def subPixelPeak(self, left, center, right):
		divisor = 2*center - right - left   #float
		return (0 if abs(divisor)<1e-3 else 0.5*(right-left)/divisor)

	def createHanningMats(self):
		hann2t, hann1t = np.ogrid[0:self._tmpl_sz[1], 0:self._tmpl_sz[0]]

		hann1t = 0.5 * (1 - np.cos(2*np.pi*hann1t/(self._tmpl_sz[0]-1)))
		hann2t = 0.5 * (1 - np.cos(2*np.pi*hann2t/(self._tmpl_sz[1]-1)))
		hann2d = hann2t * hann1t
		hann2d = hann2d.astype(np.float32)
		self.hann = [0 for i in xrange(self.numLayers)]
		for layer in xrange(self.numLayers):
			self.hann[layer] = np.tile(hann2d, (self.layer_size[layer],1,1))


	def createGaussianPeak(self, sizey, sizex):
		syh, sxh = sizey/2, sizex/2
		output_sigma = np.sqrt(sizex*sizey) / self.padding * self.output_sigma_factor
		mult = -0.5 / (output_sigma*output_sigma)
		y, x = np.ogrid[0:sizey, 0:sizex]
		y, x = (y-syh)**2, (x-sxh)**2
		res = np.exp(mult * (y+x))
		assert res.ndim==2
		return np.fft.fft2(res)

	def getFeatures(self, image, init=0, scale_adjust=None, scale_step=None):
		extracted_roi = [0,0,0,0]   #[int,int,int,int]
		cx = self._roi[0] + self._roi[2]/2  #float
		cy = self._roi[1] + self._roi[3]/2  #float

		if(init):
			if(float(self._roi[2])/float(self._roi[3]) > 2.5 ): #if(target_sz(1)/target_sz(2) > 2)
				#For objects with large height, we restrict the search window with padding.height
				padded_w = self._roi[2] * self.restrict_width
				padded_h = self._roi[3] * self.restrict_height
				#window_sz = floor(target_sz.*[1+padding.height, 1+padding.generic]);
				
			elif(float(self._roi[2]*self._roi[3])/float(image.shape[0]*image.shape[1]) >0.05 ):  #(prod(target_sz)/prod(im_sz(1:2)) > 0.05)
				# For objects with large height and width and accounting for at least 10 percent of the whole image,
				# we only search 2x height and width
				padded_w = self._roi[2] * self.restrict_large
				padded_h = self._roi[3] * self.restrict_large
				#window_sz=floor(target_sz*(1+padding.large));
				
			else:
				#otherwise, we use the padding configuration
				#window_sz = floor(target_sz * (1 + padding.generic));
				padded_w = self._roi[2] * self.padding
				padded_h = self._roi[3] * self.padding
			
			if(self.template_size > 1):
				if(padded_w >= padded_h):
					self._scale = padded_w / float(self.template_size)
				else:
					self._scale = padded_h / float(self.template_size)
				self._tmpl_sz[0] = int(padded_w / self._scale / self.cell_size)
				self._tmpl_sz[1] = int(padded_h / self._scale / self.cell_size)
			else:
				self._tmpl_sz[0] = int(padded_w / self.cell_size)
				self._tmpl_sz[1] = int(padded_h / self.cell_size)
				self._scale = 1.

			self._tmpl_sz[0] = int(self._tmpl_sz[0]) / 2 * 2
			self._tmpl_sz[1] = int(self._tmpl_sz[1]) / 2 * 2

			self.createHanningMats()

		if scale_adjust:
			extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0] * self.cell_size)
			extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1] * self.cell_size)
			extracted_roi[0] = int(cx - extracted_roi[2]/2)
			extracted_roi[1] = int(cy - extracted_roi[3]/2)

			z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
			if(z.shape[1]!=self.net_insize or z.shape[0]!=self.net_insize):
				z = cv2.resize(z, (self.net_insize, self.net_insize))

			FeaturesMap = self.get_deep_feature(z)
			#if(inithann):
			#	self.createHanningMats()  # createHanningMats need size_patch
			for layer in xrange(self.numLayers):
				#hann_window = np.tile(self.hann, (self.layer_size[layer],1,1)).transpose(1,2,0)
				FeaturesMap[layer] = self.hann[layer] * FeaturesMap[layer]
			return FeaturesMap
		elif scale_step:
			scale_sz = len(scale_step)
			times = 2 * scale_sz + 1
			imgs = [0 for i in xrange(times)]
			for i in xrange(scale_sz+1):
				if i==0:
					extracted_roi[2] = int(1. * self._scale * self._tmpl_sz[0] * self.cell_size)
					extracted_roi[3] = int(1. * self._scale * self._tmpl_sz[1] * self.cell_size)
					extracted_roi[0] = int(cx - extracted_roi[2]/2)
					extracted_roi[1] = int(cy - extracted_roi[3]/2)

					img1 = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
					if(img1.shape[1]!=self.net_insize or img1.shape[0]!=self.net_insize):
						img1 = cv2.resize(img1, (self.net_insize, self.net_insize))
					imgs[i] = img1
				else:
					extracted_roi[2] = int(1.0/scale_step[i-1] * self._scale * self._tmpl_sz[0] * self.cell_size)
					extracted_roi[3] = int(1.0/scale_step[i-1] * self._scale * self._tmpl_sz[1] * self.cell_size)
					extracted_roi[0] = int(cx - extracted_roi[2]/2)
					extracted_roi[1] = int(cy - extracted_roi[3]/2)

					img1 = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
					if(img1.shape[1]!=self.net_insize or img1.shape[0]!=self.net_insize):
						img1 = cv2.resize(img1, (self.net_insize, self.net_insize))
					imgs[2*i-1] = img1

					extracted_roi[2] = int(scale_step[i-1] * self._scale * self._tmpl_sz[0] * self.cell_size)
					extracted_roi[3] = int(scale_step[i-1] * self._scale * self._tmpl_sz[1] * self.cell_size)
					extracted_roi[0] = int(cx - extracted_roi[2]/2)
					extracted_roi[1] = int(cy - extracted_roi[3]/2)

					img2 = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
					if(img2.shape[1]!=self.net_insize or img2.shape[0]!=self.net_insize):
						img2 = cv2.resize(img2, (self.net_insize, self.net_insize))
					imgs[2*i] = img2
			
			FeaturesMaps = self.get_deep_feature(imgs, batch=True)

			for idx in xrange(times):
				for layer in xrange(self.numLayers):
					FeaturesMaps[idx][layer] = self.hann[layer] * FeaturesMaps[idx][layer]
			return FeaturesMaps
		
		else:
			raise RuntimeError("No single scale or multi scale detected.")


	def detect(self, feat):
		start = time.time()
		res = np.zeros((self._tmpl_sz[1], self._tmpl_sz[0]), np.float32)
		for layer in xrange(self.numLayers):
			cur_feat = feat[layer]
			zf = np.fft.fft2(cur_feat)
			kzf = np.sum((zf * np.conj(self._xf[layer])), axis=0) / np.prod(zf.shape)
			assert kzf.shape[0] == zf.shape[1]

			cur_res = np.real(np.fft.ifft2(self._alphaf[layer]*kzf))
			cur_res = cur_res/np.max(cur_res)
			res += cur_res * self.nweights[layer]
		
		_, pv, _, pi = cv2.minMaxLoc(res)   # pv:float  pi:tuple of int; WH format
		p = [float(pi[0]), float(pi[1])]   # cv::Point2f, [x,y]  #[float,float]

		p[0] -= res.shape[1] / 2.
		p[1] -= res.shape[0] / 2.
		
		end = time.time()
		print('detection use time : {}'.format(end-start))

		return p, pv


	def train(self, feature, train_interp_factor):
		start = time.time()

		for layer in xrange(self.numLayers):
			cur_feat = feature[layer]
			xf = np.fft.fft2(cur_feat)
			self._xf[layer] = (1-train_interp_factor)*self._xf[layer] + train_interp_factor*xf

			kf = np.sum(xf * np.conj(xf), axis=0) / np.prod(xf.shape)
			assert kf.shape[0]==xf.shape[1]

			alphaf = self._prob / (kf+self.lambdar)
			self._alphaf[layer] = (1-train_interp_factor)*self._alphaf[layer] + train_interp_factor*alphaf

		end = time.time()
		print('train use time : {}'.format(end-start))

	def init(self, image, roi):
		self._roi = map(float, roi)
		assert(roi[2]>0 and roi[3]>0)
		feat = self.getFeatures(image, init=1, scale_adjust=1.0)
		self._prob = self.createGaussianPeak(self._tmpl_sz[1], self._tmpl_sz[0])
		self._alphaf = [0 for i in xrange(self.numLayers)]
		self._xf = [np.zeros((self.layer_size[layer], self._tmpl_sz[1], self._tmpl_sz[0]), np.float32)\
																 for layer in xrange(self.numLayers)] #2layers for cv2.dft
		self.train(feat, 1.0)


	def update(self, image):
		if(self._roi[0]+self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1
		if(self._roi[1]+self._roi[3] <= 0):  self._roi[1] = -self._roi[2] + 1
		if(self._roi[0] >= image.shape[1]-1):  self._roi[0] = image.shape[1] - 2
		if(self._roi[1] >= image.shape[0]-1):  self._roi[1] = image.shape[0] - 2

		cx = self._roi[0] + self._roi[2]/2.
		cy = self._roi[1] + self._roi[3]/2.

		FeatureMaps = self.getFeatures(image, scale_step = self.scale_step)
		loc, peak_value = self.detect(FeatureMaps[0])
		res_loc, res_peak_value, res_step = loc, peak_value, 1

		for i in xrange(len(self.scale_step)):
			cur_step = self.scale_step[i]
			# Test at a smaller _scale
			new_loc1, new_peak_value1 = self.detect(FeatureMaps[2*i+1])
			# Test at a bigger _scale
			new_loc2, new_peak_value2 = self.detect(FeatureMaps[2*i+2])

			if(new_peak_value1 > new_peak_value2):
				if(new_peak_value1 * self.scale_weight[i] > res_peak_value):
					res_peak_value = new_peak_value1 * self.scale_weight[i]
					res_loc = new_loc1
					res_step = 1.0/cur_step
			else:
				if(new_peak_value2 * self.scale_weight[i] > res_peak_value):
					res_peak_value = new_peak_value2 * self.scale_weight[i]
					res_loc = new_loc2
					res_step = cur_step

		loc = res_loc
		self._scale *= res_step
		self._roi[2] *= res_step
		self._roi[3] *= res_step
		
		self._roi[0] = cx - self._roi[2]/2.0 + loc[0]*self._scale*self.cell_size
		self._roi[1] = cy - self._roi[3]/2.0 + loc[1]*self._scale*self.cell_size
		
		if(self._roi[0] >= image.shape[1]-1):  self._roi[0] = image.shape[1] - 1
		if(self._roi[1] >= image.shape[0]-1):  self._roi[1] = image.shape[0] - 1
		if(self._roi[0]+self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 2
		if(self._roi[1]+self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 2
		assert(self._roi[2]>0 and self._roi[3]>0)

		x = self.getFeatures(image, scale_adjust=1.0)
		self.train(x, self.interp_factor)

		return_box = [0., 0., 0., 0.]
		return_box[0] = max(0, min(self._roi[0], image.shape[1]-1))
		return_box[1] = max(0, min(self._roi[1], image.shape[0]-1))
		return_box[2] = max(10, min(self._roi[2], image.shape[1]-1))
		return_box[3] = max(10, min(self._roi[3], image.shape[0]-1))

		return return_box


	def get_deep_feature(self, img, batch=False):
		#pre possess of the img
		if batch==False:
			img = np.transpose(img,(2,0,1))
			img = torch.from_numpy(img).float().unsqueeze(0)
			img = img.cuda()
			features = self.net.feature_extractor(img)  #returned numpy format feature

			#post possess of the feature
			for layer in xrange(self.numLayers):
				features[layer] = torch.squeeze(features[layer], 0)
				features[layer] = features[layer].data.cpu().numpy()
				features[layer] = np.transpose(features[layer], (1, 2, 0))
				features[layer] = cv2.resize(features[layer], (self._tmpl_sz[0], self._tmpl_sz[1]), cv2.INTER_LINEAR)
				features[layer] = np.transpose(features[layer], (2, 0, 1))
			return features

		else:
			batch_sz = len(img)
			m_input = torch.empty([batch_sz, 3, self.net_insize, self.net_insize])
			for i in xrange(batch_sz):
				cur = np.transpose(img[i],(2,0,1))
				cur = torch.from_numpy(img[i]).float()
				m_input[i] = cur
			
			features = self.net.feature_extractor(m_input.cuda())
			features = features.data.cpu().numpy()

			m_return = [[0 for j in range(self.numLayers)] for i in xrange(batch_sz)]
			for layer in xrange(self.numLayers):
				for i in xrange(batch_sz):
					cur = features[layer][i]
					cur = np.transpose(cur, (1, 2, 0))
					cur = cv2.resize(cur, (self._tmpl_sz[0], self._tmpl_sz[1]), cv2.INTER_LINEAR)
					cur = np.transpose(cur, (2, 0, 1))
					m_return[i][layer] = cur
			
			return m_return





