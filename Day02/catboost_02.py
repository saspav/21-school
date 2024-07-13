import optuna
from optuna.integration import CatBoostPruningCallback

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

SEED = 17


def objective(trial: optuna.Trial) -> float:
    param = {
        "loss_function": trial.suggest_categorical("loss_function", ["binary"]),
        "auto_class_weights": trial.suggest_categorical("auto_class_weights",
                                                        [None, "Balanced"]),
        # "iterations": trial.suggest_int("iterations", 200, 2000, step=200),
        # "depth": trial.suggest_int("depth", 1, 12),
        # "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.2, step=0.05),
    }

    clf = CatBoostClassifier(cat_features=cat_features,
                             eval_metric='AUC:hints=skip_train~false',
                             #     eval_metric='TotalF1',
                             train_dir="base",
                             task_type=task_type,
                             devices=devices,
                             early_stopping_rounds=80,
                             random_seed=SEED,
                             **param)

    pruning_callback = CatBoostPruningCallback(trial, "TotalF1")

    clf.fit(X_train, Y_train,
            eval_set=[(X_test, Y_test)],
            verbose=100,
            early_stopping_rounds=80,
            callbacks=[pruning_callback],
            )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    roc_auc = roc_auc_score(Y_test, clf.predict(X_test))
    return roc_auc


start_time = print_msg('Обучение Catboost классификатор...')

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
)
study.optimize(objective, n_trials=12, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = trial.params

clf = CatBoostClassifier(**best_params)
