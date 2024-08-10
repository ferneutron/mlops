from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest",
    packages_to_install=[
        "pandas==1.3.5",
        "joblib==1.1.0",
    ],
)
def choose_best_model(
    test_dataset: Input[Dataset],
    decision_tree_model: Input[Model],
    random_forest_model: Input[Model],
    metrics: Output[Metrics],
    best_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score

    test_data = pd.read_csv(test_dataset.path)

    dt = joblib.load(decision_tree_model.path)
    rf = joblib.load(random_forest_model.path)

    dt_pred = dt.predict(test_data.drop("Species", axis=1))
    rf_pred = rf.predict(test_data.drop("Species", axis=1))

    dt_accuracy = accuracy_score(test_data["Species"], dt_pred)
    rf_accuracy = accuracy_score(test_data["Species"], rf_pred)

    metrics.log_metric("Decision Tree (Accuracy)", (dt_accuracy))
    metrics.log_metric("Random Forest (Accuracy)", (rf_accuracy))

    if dt_accuracy > rf_accuracy:
        joblib.dump(dt, best_model.path)
    else:
        joblib.dump(rf, best_model.path)
