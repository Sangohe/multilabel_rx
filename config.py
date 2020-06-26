# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., 'mydict.key = value'.

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Class names and paths to records

random_seed = 2626
exp_name = '0001-bivlab-ensemble'
class_names = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pleural Effusion']

result_dir = 'results'
train_record = '/data/DeepSARS/datasets/tf_records/CheXpert/XR_CheXpert_train_frontal_mt.tfrecord'
# valid_record = '/data/DeepSARS/datasets/tf_records/CheXpert/multiview/RXChexpert_M_valid.tfrecord'
test_record  = '/data/DeepSARS/datasets/tf_records/CheXpert/XR_CheXpert_valid_frontal_mt.tfrecord'

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Environment configuration
env = EasyDict(CUDA_VISIBLE_DEVICES='1', TF_DETERMINISTIC_OPS='1')

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Options for the network(s). network_1 is the main network and should be the only
# one you change if training with a single model. Uncomment network_2 and create more
# networks if you are going to train with an Ensemble Model.

network_1 = EasyDict(freeze=False, weights_path=None, input_shape=(224, 224, 3), n_classes=len(class_names), model_name='frontal')
network_1.model_path = None
network_1.weights_path = None

network_1.module_name = 'DenseNet121';             desc = 'densenet121';
# network_1.module_name = 'DenseNet169';             desc = 'densenet169';
# network_1.module_name = 'DenseNet201';             desc = 'densenet201';
# network_1.module_name = 'InceptionResNetV2';       desc = 'inception_resnet_v2';
# network_1.module_name = 'Xception';                desc = 'xception';

network_2 = EasyDict(freeze=False, weights_path=None, input_shape=(224, 224, 3), n_classes=len(class_names), model_name='lateral')
network_2.model_path = None
network_2.weights_path = None

network_2.module_name = 'DenseNet121';             desc += '-densenet121';
# network_2.module_name = 'DenseNet169';             desc += '-densenet169';
# network_2.module_name = 'DenseNet201';             desc += '-densenet201';
# network_2.module_name = 'InceptionResNetV2';       desc += '-inception_resnet_v2';
# network_2.module_name = 'Xception';                desc += '-xception';

# After creating all the network dictionaries, you should add all of them to `networks` 
# dictionary

networks = EasyDict(networks=[network_1, network_2])
networks.model_path = None
networks.weights_path = None

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset options
# f: frontal, l: lateral, multi: multiview, tm: template matched, ct: contidional training

# Feature dictionary to use
feature_dict = EasyDict(func='dataset.rx_chexpert', n_diseases=len(class_names))

# Dataset options
# desc += '-rx_chest14';                dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chest14_tm';             dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chexpert_f';             dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chexpert_f_ct';          dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chexpert_l';             dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chexpert_l_ct';          dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
desc += '-rx_chexpert_multi';         dataset = EasyDict(batch_size=32, shuffle=1024, prefetch=10);   dataset.map_functions = []
# desc += '-rx_chexpert_multi_ct';      dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chexpert_multi_f';       dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chexpert_multi_f_ct';    dataset = EasyDict(batch_size=8, shuffle=1024, prefetch=10);    dataset.map_functions = []
# desc += '-rx_chexpert_multi_l';       dataset = EasyDict(batch_size=32, shuffle=1024, prefetch=10);   dataset.map_functions = []
# desc += '-rx_chexpert_multi_l_ct';    dataset = EasyDict(batch_size=32, shuffle=1024, prefetch=10);   dataset.map_functions = []

## Transformations

dataset.map_functions.append('dataset.from_bytes_to_dict')
dataset.map_functions.append('dataset.extract_data_from_dict')
# dataset.map_functions.append('dataset.extract_data_from_dict_multiview')

# Single view mapping functions
# desc += '-scale_0';        dataset.map_functions.append('dataset.scale_0')
desc += '-scale_1';        dataset.map_functions.append('dataset.scale_minus1_1')
# desc += '-scale_imagenet'; dataset.map_functions.append('dataset.scale_imagenet')
desc += '-horizontal_aug'; dataset.map_functions.append('dataset.horizontal_flipping_aug')

# Multiview mapping functions
# desc += '-scale_0';        dataset.map_functions.append('dataset.scale_0_multiview')
# desc += '-scale_1';        dataset.map_functions.append('dataset.scale_minus1_1_multiview')
# desc += '-scale_imagenet'; dataset.map_functions.append('dataset.scale_imagenet_multiview')
# desc += '-horizontal_aug'; dataset.map_functions.append('dataset.horizontal_flipping_aug_multiview')

### Policies

# Single view mapping functions
# desc += '-uzeros';      dataset.map_functions.append('dataset.upolicy');           minval = 0;
# desc += '-uones';       dataset.map_functions.append('dataset.upolicy');           minval = 1;
# desc += '-lsr_uzeros';  dataset.map_functions.append('dataset.label_smoothing');   minval = 0;     maxval = .3;      
desc += '-lsr_uones';   dataset.map_functions.append('dataset.label_smoothing');   minval = .55;   maxval = .85;

# Multiview mapping functions
# desc += '-uzeros';      dataset.map_functions.append('dataset.upolicy_multiview');           minval = 0;
# desc += '-uones';       dataset.map_functions.append('dataset.upolicy_multiview');           minval = 1;
# desc += '-lsr_uzeros';  dataset.map_functions.append('dataset.label_smoothing_multiview');   minval = 0;     maxval = .3;      
# desc += '-lsr_uones';   dataset.map_functions.append('dataset.label_smoothing_multiview');   minval = .55;   maxval = .85;

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Callback options. You can choose multiple callbacks or none at all
callbacks = EasyDict()

# callbacks.decay_lr_after_epoch = EasyDict(func='callbacks.decay_lr_on_epoch_end', verbose=1)
callbacks.reduce_lr_on_plateau = EasyDict(monitor="val_loss", factor=0.1, patience=1, verbose=1, mode="min", min_lr=1e-8)
callbacks.multiple_class_auroc = EasyDict(class_names=class_names, exp_name=exp_name, stats=None)
# callbacks.model_checkpoint_callback = EasyDict(save_weights_only=False, monitor="val_loss", mode="min", save_best_only=True)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Network Training options

# train = EasyDict(func='train.train_single_network', epochs = 50)
# train = EasyDict(func='train.train_ensemble_network', epochs = 50)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Utility scripts

train = EasyDict(func='util_scripts.evaluate_multiclass_model', run_id=None, model_path="/home/santgohe/code/DeepSars/models/CheXpert/uniform/Ensemble.h5", train_record=train_record, test_record=test_record, class_names=class_names, visuals=True);
# train = EasyDict(func='util_scripts.evaluate_single_network', run_id=1, test_record=test_record, class_names=class_names);
# train = EasyDict(func='util_scripts.evaluate_late_fusion_ensemble', first_exp_id=1, second_exp_id=0, test_record=test_record, class_names=class_names, use_weighted_average=True, valid_record=valid_record)
# train = EasyDict(func='util_scripts.generate_cams', model_path="", run_id=None, image_path="", scale_func="dataset.scale_imagenet_np", class_names=class_names)