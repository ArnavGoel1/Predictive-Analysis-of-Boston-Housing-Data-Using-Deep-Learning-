import numpy as np
from tqdm import tqdm
from utils.data_preprocessing import load_and_preprocess_data
from layers.FullyConnectedLayer import FullyConnectedLayer
from layers.CNN import CNN
from layers.ReLULayer import ReLU
from layers.SquaredError import MSELoss
from utils.Optimizer import Optimizer
import matplotlib.pyplot as plt
        
                
def train(model, optimizer, trainX, trainy, loss_fct=MSELoss(), nb_epochs=1000, batch_size=32):
    training_loss = []

    for epoch in tqdm(range(nb_epochs)):
        epoch_losses = []
              
        for i in range(0, trainX.shape[0], batch_size):
            batch_end = i + batch_size
            x_batch = trainX[i:batch_end]
            target_batch = trainy[i:batch_end].reshape(-1, 1) 

            prediction = model.forward(x_batch)
            loss_value = loss_fct(prediction, target_batch)
            epoch_losses.append(loss_value)

            gradout = loss_fct.backward()
            model.backward(gradout) 

            optimizer.step() 

        
        training_loss.append(np.mean(epoch_losses))

    return training_loss
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


  
if __name__ == "__main__":
    
    trainX, testX, trainy, testy = load_and_preprocess_data('Source Code/data/HousingData.csv')

    mlp = CNN([
        FullyConnectedLayer(trainX.shape[1], 128), ReLU(),
        FullyConnectedLayer(128, 64), ReLU(),
        FullyConnectedLayer(64, 1)  
    ])
    optimizer = Optimizer(1e-3, mlp)

    loss_fct = MSELoss()

    training_loss = train(mlp, optimizer, trainX, trainy, loss_fct, nb_epochs=1000, batch_size=32)

    import matplotlib.pyplot as plt
    plt.plot(training_loss)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.show()
    final_loss = training_loss[-1]
    print(f"Final Training Loss: {final_loss:.4f}")

    test_predictions = mlp.forward(testX)
    test_predictions = mlp.forward(testX)
    final_mse = mean_squared_error(testy, test_predictions)
    print(f"Final MSE on Test Set: {final_mse:.4f}")

    final_rmse = root_mean_squared_error(testy, test_predictions)
    final_mae = mean_absolute_error(testy, test_predictions)
    print(f"Final RMSE on Test Set: {final_rmse:.4f}")
    print(f"Final MAE on Test Set: {final_mae:.4f}")

    print("\nComparing actual and predicted values:")
    for i in range(5):  
        print(f"Actual: {testy[i]:.2f} - Predicted: {test_predictions[i][0]:.2f}")