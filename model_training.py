import sklearn.metrics as metrics
from tqdm import tqdm
import torch
import numpy as np


def train(train_loader, model, loss_fn, optimizer, device):
    accuracy = []
    total_loss = 0
    predictions = []
    ratings = []
    for idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        rating = data['rating'].to(device)
        reviewer_id = data['reviewer_id'].to(device)
        product_id = data['product_id'].to(device)

        preds = model(product_id, reviewer_id)

        loss = loss_fn(preds, rating)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        predictions.extend(list(preds.cpu().detach()))
        ratings.extend(list(rating.cpu().detach()))
    rmse_loss = metrics.mean_squared_error(ratings, predictions, squared=False)
    mae_loss = metrics.mean_absolute_error(ratings, predictions)
    print(f'Training RMSE: {rmse_loss}, Training MAE: {mae_loss}')

    return rmse_loss, mae_loss, predictions, ratings


def test(test_loader, model, loss_fn, device):
    accuracy = []
    total_loss = 0
    predictions = []
    ratings = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):

            rating = data['rating'].to(device)
            reviewer_id = data['reviewer_id'].to(device)
            product_id = data['product_id'].to(device)

            preds = model(product_id, reviewer_id)
            loss = loss_fn(preds, rating)
            total_loss += loss.item()

            predictions.extend(list(preds.cpu().detach()))
            ratings.extend(list(rating.cpu().detach()))
#         rmse_loss = np.sqrt(total_loss / len(train_loader))
        rmse_loss = metrics.mean_squared_error(
            ratings, predictions, squared=False)
        mae_loss = metrics.mean_absolute_error(ratings, predictions)
        print(f'Testing RMSE: {rmse_loss}, Testing MAE: {mae_loss}')

    return rmse_loss, mae_loss, predictions, ratings


def engine(train_loader, test_loader, model, loss_fn, optimizer, epochs, device, model_name='mf'):
    best_test_loss = float('inf')
    best_test_mae = float('inf')
    best_test_predictions = None
    train_losses = []
    train_maes = []
    test_losses = []
    test_maes = []
    patience = 0
    for e in range(epochs):
        print("Starting Training ...")
        model.train()
        train_loss, train_mae, train_predictions, train_ratings = train(
            train_loader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        print("Starting Testing ...")
        test_loss, test_mae, test_predictions, test_ratings = test(
            test_loader, model, loss_fn, device)
        test_losses.append(test_loss)
        test_maes.append(test_mae)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_predictions = test_predictions
            best_test_mae = test_mae
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
            np.save(f'best_predictions_{model_name}.npy', np.array(
                best_test_predictions))
            np.save(f'ratings_{model_name}.npy', np.array(test_ratings))
            patience = 0
        else:
            patience += 1

        if patience >= 5:
            return train_losses, train_maes, test_losses, test_maes, best_test_loss, best_test_mae
    return train_losses, train_maes, test_losses, test_maes, best_test_loss, best_test_mae
