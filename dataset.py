import torch.utils.data as data
import numpy as np
from utils import process_feat, get_rgb_list_file
import torch
from torch.utils.data import DataLoader
import os  # Import os for path handling


torch.set_default_tensor_type('torch.cuda.FloatTensor') 

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.emb_folder = args.emb_folder
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.feature_size = args.feature_size
        if args.test_rgb_list is None:
            _, self.rgb_list_file = get_rgb_list_file(args.dataset, test_mode)
        else:
            self.rgb_list_file = args.test_rgb_list
    
        # Deal with different I3D feature version
        if 'v2' in self.dataset:
            self.feat_ver = 'v2'
        elif 'v3' in self.dataset:
            self.feat_ver = 'v3'
        else:
            self.feat_ver = 'v1'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list() 
        self.num_frame = 0
        self.labels = None
    

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        # ---- testing using all the dataset -
        if self.test_mode is False:  # List for training would need to be ordered from normal to abnormal
            if 'shanghai' in self.dataset:
                if self.is_normal:
                    self.list = self.list[63:]
                    print('Normal list for Shanghai Tech')
                else:
                    self.list = self.list[:63]
                    print('Abnormal list for Shanghai Tech')
            elif 'ucf' in self.dataset:
                if self.is_normal:
                    self.list = self.list[810:]
                    print('Normal list for UCF')
                else:
                    self.list = self.list[:810]
                    print('Abnormal list for UCF')
            elif 'violence' in self.dataset:
                if self.is_normal:
                    self.list = self.list[1904:]
                    print('Normal list for Violence')
                else:
                    self.list = self.list[:1904]
                    print('Abnormal list for Violence')
            elif 'ped2' in self.dataset:
                if self.is_normal:
                    self.list = self.list[6:]
                    print('Normal list for Ped2', len(self.list))
                else:
                    self.list = self.list[:6]
                    print('Abnormal list for Ped2', len(self.list))
            elif 'TE2' in self.dataset:  # Note: index starts from 0, while PyCharm line numbers start from 1
                if self.is_normal:
                    self.list = self.list[23:]
                    print('Normal list for TE2', len(self.list))
                else:
                    self.list = self.list[:23]
                    print('Abnormal list for TE2', len(self.list))
            else:
                raise Exception("Dataset undefined!!!")

    def __getitem__(self, index):
        # 0 for normal, 1 for abnormal get_label()
        label = self.get_label()  # Get video level label 0/1
        i3d_path = self.list[index].strip('\n')

        # Replace Linux-style paths with Windows-compatible paths
        i3d_path = i3d_path.replace('/', os.sep)  # Convert forward slashes to backslashes
        i3d_path = os.path.normpath(i3d_path)  # Normalize the path for Windows

        # Ensure the path is relative to the current working directory
        if not os.path.isabs(i3d_path):
            i3d_path = os.path.join(os.getcwd(), i3d_path)

        # Debug: Print the I3D path
        print(f"I3D path: {i3d_path}")

        # Check if the file exists
        if not os.path.exists(i3d_path):
            raise FileNotFoundError(f"Missing I3D feature file: {i3d_path}")

        # Load I3D features
        features = np.load(i3d_path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        


        # Construct text path dynamically
        if 'ucf' in self.dataset:
            text_path = os.path.join("save", "Crime", "sent_emb_n", i3d_path.split(os.sep)[-1][:-7] + "emb.npy")
        elif 'shanghai' in self.dataset:
            text_path = os.path.join("save", "Shanghai", "sent_emb_n", i3d_path.split(os.sep)[-1][:-7] + "emb.npy")
        elif 'violence' in self.dataset:
            text_path = os.path.join("save", "Violence", "sent_emb_n", i3d_path.split(os.sep)[-1][:-7] + "emb.npy")
        elif 'ped2' in self.dataset:
            text_path = os.path.join("save", "UCSDped2", "sent_emb_n", i3d_path.split(os.sep)[-1][:-7] + "emb.npy")
        elif 'TE2' in self.dataset:
            text_path = os.path.join("save", "TE2", "sent_emb_n", i3d_path.split(os.sep)[-1][:-7] + "emb.npy")
        else:
            raise Exception("Dataset undefined!!!")

        # Debug: Print the text path
        print(f"Text path: {text_path}")

        # Check if the text feature file exists
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Missing text feature file: {text_path}")

        # Load text features
        text_features = np.load(text_path, allow_pickle=True)
        text_features = np.array(text_features, dtype=np.float32)  # [snippet no., 768]
        min_len = min(features.shape[0], text_features.shape[0])

        
        features = features[:min_len]
        text_features = text_features[:min_len]

        if self.feature_size == 1024:
            text_features = np.tile(text_features, (5, 1, 1))  # [10, snippet no., 768]
        elif self.feature_size == 2048:
            text_features = np.tile(text_features, (10, 1, 1))  # [10, snippet no., 768]
        else:
            raise Exception("Feature size undefined!!!")
        
        if self.tranform is not None:
            features = self.tranform(features)
        
        if self.test_mode:
            text_features = text_features.transpose(1, 0, 2)  # [snippet no., 10, 768]
            return features, text_features
        else:
            # Process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [snippet no., 10, 2048] -> [10, snippet no., 2048]
            divided_features = []

            for feature in features:  # Loop 10 times
                feature = process_feat(feature, 32)  # Divide a video into 32 segments/snippets/clips
                divided_features.append(feature)
            
            divided_features = np.array(divided_features, dtype=np.float32)  # [10, 32, 2048]
         
            div_feat_text = []
            for text_feat in text_features:
                text_feat = process_feat(text_feat, 32)  # [32, 768]
                div_feat_text.append(text_feat)
            div_feat_text = np.array(div_feat_text, dtype=np.float32)
            assert divided_features.shape[1] == div_feat_text.shape[1], \
                f"Mismatch: features {divided_features.shape[1]} vs text {div_feat_text.shape[1]}"
            return divided_features, div_feat_text, label


    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label


    def __len__(self):
        return len(self.list)


    def get_num_frames(self):
        return self.num_frame


