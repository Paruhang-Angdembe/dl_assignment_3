import torch
from regression import fit_regression_model


def get_train_data(dim=1):
    """
    dim is the number of features in the input. for our purposes it will be either 1 or 2.
    """
    X_2 = torch.tensor(
        [
            [24.0, 2.0],
            [24.0, 4.0],
            [16.0, 3.0],
            [25.0, 6.0],
            [16.0, 1.0],
            [19.0, 2.0],
            [14.0, 3.0],
            [22.0, 2.0],
            [25.0, 4.0],
            [12.0, 1.0],
            [24.0, 7.0],
            [19.0, 1.0],
            [23.0, 7.0],
            [19.0, 5.0],
            [21.0, 3.0],
            [16.0, 6.0],
            [24.0, 5.0],
            [19.0, 7.0],
            [14.0, 4.0],
            [20.0, 3.0],
        ]
    )
    y = torch.tensor(
        [
            [1422.4000],
            [1469.5000],
            [1012.7000],
            [1632.2000],
            [952.2000],
            [1117.7000],
            [906.2000],
            [1307.3000],
            [1552.8000],
            [686.7000],
            [1543.4000],
            [1086.5000],
            [1495.2000],
            [1260.7000],
            [1288.1000],
            [1111.5000],
            [1523.1000],
            [1297.4000],
            [946.4000],
            [1197.1000],
        ]
    )
    if dim == 1:
        X = X_2[:, :1]
    elif dim == 2:
        X = X_2
    else:
        raise ValueError("dim must be 1 or 2")
    return X, y


def test_fit_regression_model_1d():
    X, y = get_train_data(dim=1)
    _, loss = fit_regression_model(X, y)
    print(loss)

    assert loss.item() < 4321, " loss too big"


def test_fit_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    assert loss.item() < 400


def test_fit_and_predict_regression_model_1d():
    X, y = get_train_data(dim=1)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[20.0], [15.0], [10.0]])
    y_pred = model(X_test)

    assert isinstance(y_pred, torch.Tensor), "y_pred is not a Tensor"
    assert (
        torch.abs(y_pred - torch.tensor([[1252.3008], [939.9971], [627.6935]])) < 2
    ).all(), f" y_pred is not correct"
    assert y_pred.shape == (3, 1), " y_pred shape is not correct"


def test_fit_and_predict_regression_model_2d():
    X, y = get_train_data(dim=2)
    model, loss = fit_regression_model(X, y)
    X_test = torch.tensor([[20.0, 2.0], [15.0, 3.0], [10.0, 4.0]])
    y_pred = model(X_test)

    assert (
        torch.abs(y_pred - torch.tensor([[1191.9037], [943.9369], [695.9700]])) < 2
    ).all(), " y_pred is not correct"
    assert y_pred.shape == (3, 1), " y_pred shape is not correct"


# if __name__ == "__main__":
#     test_fit_regression_model_1d()
#     test_fit_regression_model_2d()
#     test_fit_and_predict_regression_model_1d()
#     test_fit_and_predict_regression_model_2d()
