# ------------------------------------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., 'mydict.key = value'.

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

# ------------------------------------------------------------------------------------------------------
# Paths, image details and batch size

n_classes = 8
channels = 3
img_size = 224
initial_lr = 1e-3
random_seed = 2626

result_dir = 'results'
data_dir = '/data/DeepSARS/datasets/tf_records/ChestX-Ray14/raw/'
train_record = '/data/DeepSARS/datasets/tf_records/CheXpert/XR_CheXpert_train_frontal_mt.tfrecord'
valid_record = '/data/DeepSARS/datasets/tf_records/CheXpert/XR_CheXpert_valid_frontal_mt.tfrecord'
test_record = None

# ------------------------------------------------------------------------------------------------------
# Training configuration

env          = EasyDict(CUDA_VISIBLE_DEVICES='0')                         # Enviroment variables
feature_dict = EasyDict(func='dataset.rx_chest14', n_diseases=n_classes)  # Options for dataset func.
train        = EasyDict(func='train.conditional_train')                   # Options for main training func.
network      = EasyDict(func='networks.Generator')                        # Options for the network.

# ------------------------------------------------------------------------------------------------------
# Choose which network to use

network = 'densenet121';             desc = 'densenet121';
# network = 'densenet169';             desc = 'densenet169';
# network = 'densenet201';             desc = 'densenet201';
# network = 'inception_resnet_v2';     desc = 'inception_resnet_v2';
# network = 'xception';                desc = 'xception';

# ------------------------------------------------------------------------------------------------------
# Dataset (choose one)
# f: frontal, l: lateral, multi: multiview, tm: template matched, ct: contidional training

# desc += '-rx_chest14';            dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chest14_tm';         dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
desc += '-rx_chexpert_f';         dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_f_ct';      dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_l';         dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_l_ct';      dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_multi';     dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_multi_ct';  dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []

# ------------------------------------------------------------------------------------------------------
# Transformations

# Always keep these two functions uncommented
dataset.map_functions.append('dataset.from_bytes_to_dict')
dataset.map_functions.append('dataset.extract_data_from_dict')

# desc += '-scale_0';        dataset.map_functions.append('dataset.scale_0')
# desc += '-scale_1';        dataset.map_functions.append('dataset.scale_minus1_1')
desc += '-scale_imagenet'; dataset.map_functions.append('dataset.scale_imagenet')
desc += '-horizontal_aug'; dataset.map_functions.append('dataset.horizontal_flipping_aug')

# ------------------------------------------------------------------------------------------------------
# Choose one policy

# desc += '-uzeros';      dataset.map_functions.append('dataset.upolicy');           minval = 0;
# desc += '-uones';       dataset.map_functions.append('dataset.upolicy');           minval = 1;
# desc += '-lsr_uzeros';  dataset.map_functions.append('dataset.label_smoothing');   minval = 0;     maxval = .3;      
desc += '-lsr_uones';   dataset.map_functions.append('dataset.label_smoothing');   minval = .55;   maxval = .85;

# ------------------------------------------------------------------------------------------------------
# Utility scripts

# train = EasyDict(func='util_scripts.evaluate', run_id=0);   desc = 'evaluate'
# train = EasyDict(func='util_scripts.ensemble', run_id=0);   desc = 'ensemble'