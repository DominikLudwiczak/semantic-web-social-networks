from ntscraper import Nitter
import pandas as pd
from multiprocessing import freeze_support


def main():
    scraper = Nitter(log_level=1, skip_instance_check=False)
    data = pd.read_csv("midterm-2018/data.csv")
    data = data[["user_id", "screen_name", "name"]]
    sample = data.sample(10)

    for index, row in sample.iterrows():
        tweets = scraper.get_tweets(
            ["4858296837", "ncaraballoPR", "Natalie Caraballo"],
            mode="user",
            number=1,
        )
        print(tweets)
    print("Done")


if __name__ == "__main__":
    freeze_support()
    main()
