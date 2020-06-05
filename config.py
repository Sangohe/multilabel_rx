# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., 'mydict.key = value'.

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Class names and paths

random_seed = 2626
class_names = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pleural Effusion']

result_dir = 'results'
data_dir = '/data/DeepSARS/datasets/tf_records/ChestX-Ray14/raw/'
train_record = '/data/DeepSARS/datasets/tf_records/CheXpert/multiview/RXChexpert_MF_train.tfrecord'
valid_record = '/data/DeepSARS/datasets/tf_records/CheXpert/multiview/RXChexpert_MF_valid.tfrecord'
test_record = None

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Execution configuration

env          = EasyDict(CUDA_VISIBLE_DEVICES='1')                                # Enviroment variables
feature_dict = EasyDict(func='dataset.rx_chexpert', n_diseases=len(class_names)) # Options for dataset func.
train        = EasyDict(func='train.train_single_network')                       # Options for main training func.
network      = EasyDict()                                                        # Options for the network
callbacks    = EasyDict()                                                        # Callbacks options

train.epochs = 25;   train.initial_lr = 1e-3;   train.verbose = 2

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which network to use

network.model_name = 'DenseNet121';             desc = 'densenet121';
# network.model_name = 'DenseNet169';             desc = 'densenet169';
# network.model_name = 'DenseNet201';             desc = 'densenet201';
# network.model_name = 'InceptionResNetV2';       desc = 'inception_resnet_v2';
# network.model_name = 'Xception';                desc = 'xception';

network.input_shape = (224, 224, 3)
network.n_classes = len(class_names)   
network.weights_path = None
network.freeze = False

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset (choose one)
# f: frontal, l: lateral, multi: multiview, tm: template matched, ct: contidional training

# desc += '-rx_chest14';            dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chest14_tm';         dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_f';         dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_f_ct';      dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_l';         dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_l_ct';      dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_multi';     dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_multi_ct';  dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []
desc += '-rx_chexpert_multi_l';   dataset = EasyDict(batch_size=32, shuffle=1024, prefetch=10);  dataset.map_functions = []
# desc += '-rx_chexpert_multi_f';   dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);  dataset.map_functions = []

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Transformations

# Always keep these two functions uncommented
dataset.map_functions.append('dataset.from_bytes_to_dict')
dataset.map_functions.append('dataset.extract_data_from_dict')

# desc += '-scale_0';        dataset.map_functions.append('dataset.scale_0')
# desc += '-scale_1';        dataset.map_functions.append('dataset.scale_minus1_1')
desc += '-scale_imagenet'; dataset.map_functions.append('dataset.scale_imagenet')
desc += '-horizontal_aug'; dataset.map_functions.append('dataset.horizontal_flipping_aug')

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose one policy

# desc += '-uzeros';      dataset.map_functions.append('dataset.upolicy');           minval = 0;
# desc += '-uones';       dataset.map_functions.append('dataset.upolicy');           minval = 1;
# desc += '-lsr_uzeros';  dataset.map_functions.append('dataset.label_smoothing');   minval = 0;     maxval = .3;      
desc += '-lsr_uones';   dataset.map_functions.append('dataset.label_smoothing');   minval = .55;   maxval = .85;

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose callbacks

# callbacks.decay_lr_after_epoch      = EasyDict(func='callbacks.decay_lr_on_epoch_end', verbose=1)
callbacks.reduce_lr_on_plateau      = EasyDict(monitor="val_loss", factor=0.1, patience=1, verbose=1, mode="min", min_lr=1e-8)
callbacks.model_checkpoint_callback = EasyDict(save_weights_only=False, monitor="val_loss", mode="min", save_best_only=True)
callbacks.multiple_class_auroc      = EasyDict(class_names=class_names, stats=None)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Utility scripts

# train = EasyDict(func='util_scripts.evaluate', run_id=0);   desc = 'evaluate'
# train = EasyDict(func='util_scripts.ensemble', run_id=0);   desc = 'ensemble'