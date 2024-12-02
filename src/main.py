from data_manager import DataManager
from train import Train


def main():
    data_manager = DataManager()
    train = Train(data_manager)
    train.train()


if __name__ == "__main__":
    main()
