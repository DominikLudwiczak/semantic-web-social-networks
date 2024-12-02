from twikit import Client
import json
from dotenv import load_dotenv
import os
import asyncio
import pandas as pd
from IPython.display import display


async def main():
    load_dotenv()
    AUTH_INFO_1 = os.getenv("AUTH_INFO_1")
    PASSWORD = os.getenv("PASSWORD")
    if PASSWORD == None:
        print("Make sure to setup login credentials in .env file")
        return
    client = Client(language="en-US")
    if not os.path.exists("cookies.json"):
        try:
            await client.login(auth_info_1=AUTH_INFO_1, password=PASSWORD)
            client.save_cookies("cookies.json")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

    client.load_cookies(path="cookies.json")

    user = await client.get_user_by_screen_name("vivaliteracy")
    tweets = await user.get_tweets("Tweets", count=5)

    tweets_to_store = []
    for tweet in tweets:
        tweets_to_store.append(
            {
                "created_at": tweet.created_at,
                "favorite_count": tweet.favorite_count,
                "full_text": tweet.full_text,
            }
        )

    df = pd.DataFrame(tweets_to_store)
    df = df.sort_values(by="favorite_count", ascending=False)
    display(df)

    return df


if __name__ == "__main__":
    asyncio.run(main())
