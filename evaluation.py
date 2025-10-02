import numpy as np
import keras.ops as ops


def test_model_batch(model, model_0_sampler, model_1_sampler, threshold, n, n_norm_factor, sample_size=1000, pr=True):
    num_samples_per_class = sample_size // 2

    data_0 = model_0_sampler(num_samples_per_class, n=n)
    data_1 = model_1_sampler(num_samples_per_class, n=n)

    X_test = np.concatenate([data_0, data_1], axis=0)[:, :, None]
    n_test = np.full((sample_size, 1), (n / n_norm_factor), dtype=np.float32)

    y_true = np.concatenate([np.zeros((num_samples_per_class, 1)), np.ones(
        (num_samples_per_class, 1))], axis=0)

    y_pred = model((X_test, n_test), training=False)

    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]

    pred_labels = ops.cast(y_pred > threshold, "float32")
    y_true_tensor = ops.convert_to_tensor(y_true, dtype="float32")

    is_class_0 = ops.equal(y_true_tensor, 0.0)
    is_class_1 = ops.equal(y_true_tensor, 1.0)
    correct_preds = ops.equal(pred_labels, y_true_tensor)

    correct_0 = ops.sum(ops.cast(ops.logical_and(
        correct_preds, is_class_0), "float32"))
    correct_1 = ops.sum(ops.cast(ops.logical_and(
        correct_preds, is_class_1), "float32"))

    acc_0 = correct_0 / num_samples_per_class
    acc_1 = correct_1 / num_samples_per_class

    if pr:
        correct_0_np = int(ops.convert_to_numpy(correct_0))
        correct_1_np = int(ops.convert_to_numpy(correct_1))
        print(
            f"  Model 0 accuracy: {ops.convert_to_numpy(acc_0)*100:.2f}% ({correct_0_np}/{num_samples_per_class})")
        print(
            f"  Model 1 accuracy: {ops.convert_to_numpy(acc_1)*100:.2f}% ({correct_1_np}/{num_samples_per_class})")

    return ops.convert_to_numpy(acc_0), ops.convert_to_numpy(acc_1)
