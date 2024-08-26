#!/bin/bash

optuna_params_1() {
    export d_model=95
    export e_layers=6
    export learning_rate=0.008217905449200933
    export batch_size=114
}

optuna_params_2() {
    export d_model=30
    export e_layers=4
    export learning_rate=0.0013750896962487524
    export batch_size=95
}

optuna_params_3() {
    export d_model=94
    export e_layers=5
    export learning_rate=0.009198733622505206
    export batch_size=91
    export best_vali_loss=0.8416457772254944
    export num_trial=100
    export pred_len=5
}

optuna_params_4() {
    export d_model=89
    export e_layers=6
    export learning_rate=0.00927494654492881
    export batch_size=98
    export best_vali_loss=0.7869106531143188
    export num_trial=100
    export pred_len=5
}

optuna_params_5() {
    export d_model=97
    export e_layers=6
    export learning_rate=0.008864538507400333
    export batch_size=106
    export best_vali_loss=0.810820460319519
    export num_trial=100
    export pred_len=7
}

optuna_params_6() {
    export d_model=53
    export e_layers=5
    export learning_rate=0.009794730425623536
    export batch_size=98
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8591528534889221
    export num_trial=200
    export pred_len=9
}


optuna_params_40() {
    export seq_len=150
    export learning_rate=0.007786789952406314
    export batch_size=73
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.9361312985420227
    export num_trial=100
    export pred_len=7
    export model="DLinear"
}

optuna_params_41() {
    export seq_len=113
    export learning_rate=0.008446224459979099
    export batch_size=85
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.8920356035232544
    export num_trial=100
    export pred_len=9
    export model="DLinear"
}

optuna_params_42() {
    export d_model=91
    export e_layers=6
    export learning_rate=0.008275042722071932
    export batch_size=102
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8740413188934326
    export num_trial=100
    export pred_len=15
    export model="iTransformer"
}

optuna_params_43() {
    export e_layers=4
    export learning_rate=0.0025104253535677824
    export batch_size=76
    export data_path="early_variables_for_mine_price.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.714719295501709
    export num_trial=100
    export pred_len=3
    export model="PatchTST"
}
optuna_params_44() {
    export seq_len=97
    export learning_rate=0.005714417677461466
    export batch_size=84
    export data_path="early_variables_for_mine_price_5500K0.8S.csv"
    export target="5500K0.8S"
    export best_vali_loss=0.8050065040588379
    export num_trial=100
    export pred_len=10
    export model="DLinear"
}

optuna_params_45() {
    export seq_len=98
    export learning_rate=0.00928171584263674
    export batch_size=89
    export data_path="early_variables_for_mine_price_5500K0.8S.csv"
    export target="5500K0.8S"
    export best_vali_loss=0.6323685050010681
    export num_trial=100
    export pred_len=5
    export model="DLinear"
}

optuna_params_46() {
    export e_layers=3
    export learning_rate=0.005730043813346062
    export batch_size=79
    export data_path="early_variables_for_mine_price.csv"
    export target="5500K0.8S"
    export best_vali_loss=0.7815828919410706
    export num_trial=100
    export pred_len=15
    export model="PatchTST"
}

optuna_params_47() {
    export e_layers=5
    export learning_rate=0.002119350340616355
    export batch_size=77
    export data_path="early_variables_for_mine_price_5500K0.8S.csv"
    export target="5500K0.8S"
    export best_vali_loss=0.6389580368995667
    export num_trial=100
    export pred_len=5
    export model="PatchTST"
}
optuna_params_48() {
    export e_layers=3
    export learning_rate=0.008213189045416576
    export batch_size=77
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8219729661941528
    export num_trial=100
    export pred_len=15
    export model="PatchTST"
}



optuna_params_50() {
    export e_layers=5
    export learning_rate=0.00410964828073033
    export batch_size=96
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.9776871800422668
    export num_trial=100
    export pred_len=30
    export model="PatchTST"
}

optuna_params_51() {
    export e_layers=3
    export learning_rate=0.00606287867111759
    export batch_size=82
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8199081420898438
    export num_trial=100
    export pred_len=15
    export model="PatchTST"
}
