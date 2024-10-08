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

optuna_params_52() {
    export e_layers=4
    export learning_rate=0.0051906702684410295
    export batch_size=68
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8472750782966614
    export num_trial=100
    export pred_len=20
    export model="PatchTST"
}

optuna_params_53() {
    export e_layers=3
    export learning_rate=0.00636539274270603
    export batch_size=64
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.7812607884407043
    export num_trial=100
    export pred_len=20
    export model="PatchTST"
}

optuna_params_54() {
    export e_layers=5
    export learning_rate=0.0023764229677456054
    export batch_size=69
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.7906928062438965
    export num_trial=60
    export pred_len=5
    export model="PatchTST"
}



optuna_params_55() {
    export e_layers=3
    export learning_rate=0.7906928062438965
    export batch_size=69
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.7906928062438965
    export num_trial=60
    export pred_len=20
    export model="PatchTST"
}
optuna_params_56() {
    export e_layers=3
    export learning_rate=0.009915613624835899
    export batch_size=68
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=1.1062813997268677
    export num_trial=100
    export pred_len=20
    export model="PatchTST"
}

optuna_params_57() {
    export e_layers=3
    export learning_rate=0.004780459442660464
    export batch_size=67
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.9738560914993286
    export num_trial=100
    export pred_len=20
    export model="PatchTST"
}


optuna_params_58() {
    export seq_len=107
    export learning_rate=0.008519535712067818
    export batch_size=87
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=1.0215210914611816
    export num_trial=100
    export pred_len=20
    export model="DLinear"
}

optuna_params_59() {
    export d_model=80
    export e_layers=6
    export learning_rate=0.005940592988430431
    export batch_size=73
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=1.0374822616577148
    export num_trial=100
    export pred_len=20
    export model="iTransformer"
}

optuna_params_60() {
    export seq_len=137
    export learning_rate=0.006110457034631447
    export batch_size=78
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=1.0925967693328857
    export num_trial=100
    export pred_len=15
    export model="DLinear"
}

optuna_params_61() {
    export d_model=14
    export e_layers=3
    export learning_rate=0.0024515679073350287
    export batch_size=70
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.9837453365325928
    export num_trial=100
    export pred_len=15
    export model="iTransformer"
}

optuna_params_62() {
    export d_model=76
    export e_layers=5
    export learning_rate=0.007040465444811926
    export batch_size=75
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.9701333045959473
    export num_trial=100
    export pred_len=10
    export model="iTransformer"
}

optuna_params_63() {
    export d_model=44
    export e_layers=4
    export learning_rate=0.0076180813144990835
    export batch_size=64
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=0.8030768632888794
    export num_trial=100
    export pred_len=5
    export model="iTransformer"
}

optuna_params_64() {
    export d_model=19
    export e_layers=2
    export learning_rate=0.0016913654089392787
    export batch_size=107
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=1.078775405883789
    export num_trial=100
    export pred_len=20
    export model="iTransformer"
}


optuna_params_65() {
    export e_layers=4
    export learning_rate=0.0077775645989396425
    export batch_size=66
    export data_path="early_variables_for_mine_price_5000K0.8S.csv"
    export target="5000K0.8S"
    export best_vali_loss=1.0046721696853638
    export num_trial=100
    export pred_len=20
    export model="PatchTST"
}
