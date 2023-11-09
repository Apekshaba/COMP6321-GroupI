import yaml
from train.train import train_model
from data.dataset import get_data_loaders

if __name__ == '__main__':
    hyperparameter_files = ["config/hyperparameters/hyperparameters1.yaml", "config/hyperparameters/hyperparameters2.yaml"]
    training_data_dir = 'C:/Users/15148/Desktop/Fall2023/ML/Project/dataset/test'
    validation_data_dir = 'C:/Users/15148/Desktop/Fall2023/ML/Project/dataset/validation'

    validation_data_loader = get_data_loaders(validation_data_dir, batch_size=32)

    for hp_file in hyperparameter_files:
        with open(hp_file, 'r') as file:
            hyperparameters = yaml.safe_load(file)

        save_path = f'model_lr{hyperparameters["learning_rate"]}.pt'

        train_model(training_data_dir, hyperparameters["num_epochs"], hyperparameters["batch_size"], hyperparameters["learning_rate"], save_path, validation_data_loader)
