from src.data_manager import DataManager
from src.train import Train


def main():
    data_manager = DataManager()
    train_dataloader = data_manager.get_train_dataloader()
    test_dataloader = data_manager.get_test_dataloader()
    train = Train(train_dataloader, test_dataloader)
    train.train()

if __name__ == "__main__":
    main()
