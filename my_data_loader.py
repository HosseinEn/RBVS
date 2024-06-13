import os
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
		self.root = root
		self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root)]
		self.image_size = image_size
		self.mode = mode
		self.rotation_degrees = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		print(f"Image count in {self.mode} path: {len(self.image_paths)}")

	def get_paths(self, index):
		image_path = self.image_paths[index]
		filename = image_path.split('_')[-1][:-len(".jpg")]
		GT_path = self.GT_paths + 'Image_' + filename + '_1stHO.png'
		return image_path, GT_path

	def aspect_ratio(self, image):
		return image.size[0] / image.size[1]
	
	def train_mode(self):
		return self.mode == 'train'

	def __getitem__(self, index):
		image_path, GT_path = self.get_paths(index)
		image = Image.open(image_path)
		GT = Image.open(GT_path)

		aspect_ratio = self.aspect_ratio(image)

		transform_list = []

		resize_range = random.randint(300,320)
		transform_list.append(T.Resize((int(resize_range*aspect_ratio),resize_range)))

		if self.train_mode() and random.random() <= self.augmentation_prob:
			rotation_degree = random.choice(self.rotation_degrees)

			if rotation_degree in [90,270]:
				aspect_ratio = 1 / aspect_ratio

			transform_list.append(T.RandomRotation(rotation_degree))
						
			transform_list.append(T.RandomRotation(random.randint(-10,10)))
			crop_range = random.randint(250,270)
			transform_list.append(T.CenterCrop((int(crop_range*aspect_ratio),crop_range)))
			transform = T.Compose(transform_list)
			
			image = transform(image)
			GT = transform(GT)

			shift_left = random.randint(0,20)
			shift_upper = random.randint(0,20)
			shift_right = image.size[0] - random.randint(0,20)
			shift_lower = image.size[1] - random.randint(0,20)
			image = image.crop(box=(shift_left,shift_upper,shift_right,shift_lower))
			GT = GT.crop(box=(shift_left,shift_upper,shift_right,shift_lower))

			if random.random() < 0.5:
				image = F.hflip(image)
				GT = F.hflip(GT)

			if random.random() < 0.5:
				image = F.vflip(image)
				GT = F.vflip(GT)

			color_jitter = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)
			image = color_jitter(image)


		# Final resize, convert to tensor, and normalize
		final_transform = T.Compose([
			T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)),
			T.ToTensor(),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])


		image = final_transform(image)
		GT = final_transform(GT)

		return image, GT

	def __len__(self):
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
