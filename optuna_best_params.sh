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


optuna_params_7() {
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=6.26425313949585
    export num_trial=2
    export pred_len=5
}

optuna_params_8() {
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.7863596677780151
    export num_trial=2
    export pred_len=3
}

optuna_params_9() {
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.6254637241363525
    export num_trial=2
    export pred_len=5
}

optuna_params_10() {
    export data_path="early_variables_for_mine_price_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.4489376544952393
    export num_trial=2
    export pred_len=7
    export model="iTransformer"
}



optuna_params_11() {
    export d_model=198
    export e_layers=3
    export learning_rate=0.0024703334916285546
    export batch_size=117
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8915119171142578
    export num_trial=2
    export pred_len=5
    export model="PatchTST"
}

optuna_params_12() {
    export e_layers=5
    export learning_rate=0.005132156726697998
    export batch_size=98
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.9328376650810242
    export num_trial=2
    export pred_len=11
    export model="PatchTST"
}

optuna_params_13() {
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.0003293345507699996
    export num_trial=2
    export pred_len=11
    export model="PatchTST"
}

optuna_params_14() {
    export e_layers=5
    export learning_rate=0.005238661147253915
    export batch_size=70
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.0003012297092936933
    export num_trial=2
    export pred_len=11
    export model="PatchTST"
}

optuna_params_15() {
    export e_layers=3
    export learning_rate=0.005339948092918917
    export batch_size=79
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8632772564888
    export num_trial=2
    export pred_len=15
    export model="PatchTST"
}

optuna_params_16() {
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.1691378355026245
    export num_trial=2
    export pred_len=13
    export model="TimesNet"
}

optuna_params_17() {
    export e_layers=4
    export learning_rate=0.00496051169357966
    export batch_size=90
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.0251514911651611
    export num_trial=2
    export pred_len=15
    export model="PatchTST"
}

optuna_params_18() {
    export e_layers=3
    export learning_rate=0.008229273178029205
    export batch_size=82
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8617041707038879
    export num_trial=5
    export pred_len=15
    export model="PatchTST"
}

optuna_params_19() {
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.9232722520828247
    export num_trial=2
    export pred_len=20
    export model="TimesNet"
}

optuna_params_20() {
    export e_layers=4
    export learning_rate=0.00540885908003581
    export batch_size=98
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.9218301177024841
    export num_trial=5
    export pred_len=10
    export model="PatchTST"
}

optuna_params_21() {
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.047999620437622
    export num_trial=2
    export pred_len=15
    export model="TimesNet"
}

optuna_params_22() {
    export e_layers=4
    export learning_rate=0.006706950185169397
    export batch_size=68
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=0.8774958252906799
    export num_trial=5
    export pred_len=5
    export model="PatchTST"
}

optuna_params_23() {
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.0475199222564697
    export num_trial=2
    export pred_len=10
    export model="TimesNet"
}

optuna_params_24() {
    export data_path="runmin_an_factors_4500K1.0S.csv"
    export target="4500K1.0S"
    export best_vali_loss=1.0529619455337524
    export num_trial=2
    export pred_len=5
    export model="TimesNet"
}
