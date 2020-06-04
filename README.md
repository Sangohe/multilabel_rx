# DeepSars - RX Multiclass classification 

## Desired folder structure for every experiment subfolder

```
    scripts
    README.md
    results
    │ 
    └───experiment-subfolder1
    │   └───README.md/txt -> training
    │   └─── log.txt -> training
    │   └─── config.txt -> training
    │   └─── best_auc_model.h5/weights -> training
    │   └─── best_loss_model.h5/weights -> training
    │   └─── 0001_BivL2abNamingConvention.h5 -> training
    │   └─── roc_curve.png -> util_scripts.evaluate_model
    │   └─── metrics_on_evaluation.pkl -> util_scripts.evaluate_model
    └───experiment-subfolder2
        └─── README.md/txt -> training
        └─── log.txt -> training
        └─── config.txt -> training
        └─── ...
```

## To Do list
* Add a `README.md` in the experiment subfolder with information about the training 
* Create `util_scripts.py` with the following functions:
  * `evaluate_on_set:` takes a trained network and a trained dataset to generate a pickle with metrics and save the ROC curve for the given model
* Modify the model save in `train_single_network` to match the naming convention proposed in the group
* Write a better `README.md`