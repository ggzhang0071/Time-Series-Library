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

