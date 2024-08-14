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
    export d_model=86
    export e_layers=5
    export learning_rate=0.008973173449418436
    export batch_size=71
}

optuna_params_4() {
    export best_vali_loss=1.970313549041748
    export num_trial=1
    export pred_len=5
}

optuna_params_5() {
    export best_vali_loss=1.970313549041748
    export num_trial=1
    export pred_len=5
}
