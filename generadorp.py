from mpi4py import MPI
import os
import json
import bz2
import getopt
import sys
import time
import argparse
import networkx as nx
from datetime import datetime
from collections import defaultdict
from itertools import combinations



def is_tweet_valid(tweet, start_date, end_date, hashtags):
    tweet_date_str = tweet.get('created_at')
    if tweet_date_str:
        tweet_date = datetime.strptime(tweet_date_str, '%a %b %d %H:%M:%S +0000 %Y').date()
    else:
        return False

    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    if not (start_date <= tweet_date <= end_date):
        return False

    if hashtags:
        tweet_hashtags = {hashtag['text'].lower() for hashtag in tweet.get('entities', {}).get('hashtags', [])}
        if not tweet_hashtags.intersection(hashtags):
            return False

    return True

def process_file(file_path, start_date, end_date, hashtags):
    """
    Process a single .json.bz2 file and return a list of valid tweets.
    """
    tweets = []
    with bz2.open(file_path, "rt") as f:  # 'rt' mode for text reading
        for line in f:
            tweet = process_line(line, start_date, end_date, hashtags)
            if tweet is not None:
                tweets.append(tweet)
    return tweets

def process_line(line, start_date, end_date, hashtags):
    """
    Process a single line (a single tweet) and return the tweet if it's valid, or None otherwise.
    """
    try:
        tweet = json.loads(line.strip())
        if is_tweet_valid(tweet, start_date, end_date, hashtags):
            return tweet
    except json.JSONDecodeError:
        return None


def get_tweets(directory, start_date, end_date, hashtags):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json.bz2"):
                all_files.append(os.path.join(root, file))

    files_per_process = len(all_files) // size
    start_index = rank * files_per_process
    end_index = start_index + files_per_process if rank != size - 1 else len(all_files)

    tweets = []
    for file_index in range(start_index, end_index):
        file_path = all_files[file_index]
        tweets.extend(process_file(file_path, start_date, end_date, hashtags))

    all_tweets = comm.gather(tweets, root=0)

    if rank == 0:
        combined_tweets = [tweet for process_tweets in all_tweets for tweet in process_tweets]
        return combined_tweets
    else:
        return []

def add_node_to_graph(graph, node):
    if node != "null" and not graph.has_node(node):
        graph.add_node(node)

def add_or_update_edge(graph, node1, node2):
    if not graph.has_edge(node1, node2):
        graph.add_edge(node1, node2, weight=0)
    graph[node1][node2]["weight"] += 1

def process_user_tweet(tweet):
    user_screen_name = tweet["user"]["screen_name"] if 'user' in tweet else None
    retweeted = "retweeted_status" in tweet
    original_tweet = tweet["retweeted_status"] if retweeted else None
    original_user_screen_name = original_tweet["user"]["screen_name"] if original_tweet else None

    return user_screen_name, retweeted, original_tweet, original_user_screen_name
def create_retweet_graph(tweets):
    graph = nx.DiGraph()

    for tweet in tweets:
        user_screen_name, retweeted, _, original_user_screen_name = process_user_tweet(tweet)

        if retweeted:
            add_node_to_graph(graph, user_screen_name)
            add_node_to_graph(graph, original_user_screen_name)
            add_or_update_edge(graph, user_screen_name, original_user_screen_name)

    return graph
def update_retweet_json_data(retweets_data, user_screen_name, original_tweet, original_user_screen_name):
    original_tweet_id = original_tweet["id_str"]
    if original_user_screen_name not in retweets_data:
        retweets_data[original_user_screen_name] = {
            'username': original_user_screen_name, 
            "receivedRetweets": 0,
            "tweets": {}
        }

    if original_tweet_id not in retweets_data[original_user_screen_name]["tweets"]:
        retweets_data[original_user_screen_name]["tweets"][original_tweet_id] = {
            "retweetedBy": []
        }

    if user_screen_name not in retweets_data[original_user_screen_name]["tweets"][original_tweet_id]["retweetedBy"]:
        retweets_data[original_user_screen_name]["tweets"][original_tweet_id]["retweetedBy"].append(user_screen_name)
        retweets_data[original_user_screen_name]["receivedRetweets"] += 1

def create_retweet_json(tweets):
    retweets_data = {}

    for tweet in tweets:
        user_screen_name, retweeted, original_tweet, original_user_screen_name = process_user_tweet(tweet)

        if retweeted:
            update_retweet_json_data(retweets_data, user_screen_name, original_tweet, original_user_screen_name)

    sorted_retweets = sorted(retweets_data.values(), key=lambda x: x['receivedRetweets'], reverse=True)
    return {'retweets': sorted_retweets}

def create_mention_graph(tweets):
    graph = nx.DiGraph()

    for tweet in tweets:
        user_screen_name, retweeted, _, _ = process_user_tweet(tweet)
        if not retweeted and user_screen_name:
            mentioned_users = {mention["screen_name"] for mention in tweet.get("entities", {}).get("user_mentions", []) if mention["screen_name"] != "null"}

            add_node_to_graph(graph, user_screen_name)
            for mentioned_user in mentioned_users:
                add_node_to_graph(graph, mentioned_user)
                add_or_update_edge(graph, user_screen_name, mentioned_user)

    return graph

def create_coretweet_json(tweets):
    retweet_dict = {}
    coretweets = []
    index_guide = {}
    index = 0

    for tweet in tweets:
        retweeter = tweet['user']['screen_name']

        # Check if the tweet is a retweet
        if 'retweeted_status' in tweet and 'user' in tweet:
            author = tweet['retweeted_status']['user']['screen_name']
            if author != retweeter and author != "null" and retweeter != "null":
                # Update the retweet dictionary
                retweet_dict.setdefault(retweeter, []).append(author)

    result = {}
    for key, authors_list in retweet_dict.items():
        seen_authors = set()
        for author in authors_list:
            if author not in seen_authors and author != key:
                seen_authors.add(author)

        # Store the pair in the dictionary
        for combo in combinations(seen_authors, 2):
            author_pair = tuple(sorted(combo))
            if author_pair not in result:
                result[author_pair] = {
                    'authors': {'u1': author_pair[0], 'u2': author_pair[1]},
                    'totalCoretweets': 0,
                    'retweeters': set()
                }
                index_guide[author_pair] = index
                index += 1
                coretweets.append(result[author_pair])

            result[author_pair]['retweeters'].add(key)
            result[author_pair]['totalCoretweets'] += 1

    # Updating the coretweets list with the final count and retweeters
    for author_pair, data in result.items():
        coretweets[index_guide[author_pair]]['retweeters'] = list(data['retweeters'])

    # Sort the coretweets list by the total number of coretweets
    sorted_coretweets = sorted(coretweets, key=lambda x: x['totalCoretweets'], reverse=True)

    return {'coretweets': sorted_coretweets}





def create_coretweet_graph(tweets):
    graph = nx.Graph()

    for tweet in tweets:
        user_screen_name, retweeted, original_tweet, _ = process_user_tweet(tweet)
        if retweeted and user_screen_name and original_tweet:
            author = original_tweet["user"]["screen_name"]
            if author != "null" and user_screen_name != "null" and author != user_screen_name:
                add_node_to_graph(graph, author)
                add_node_to_graph(graph, user_screen_name)
                add_or_update_edge(graph, author, user_screen_name)

    return graph


def update_mention_json_data(mentions_data, user_screen_name, mentioned_user, tweet_id):
    if mentioned_user not in mentions_data:
        mentions_data[mentioned_user] = {
            "username": mentioned_user,
            "receivedMentions": 0,
            "mentions": []
        }

    existing_mention = None
    for mention in mentions_data[mentioned_user]["mentions"]:
        if mention["mentionBy"] == user_screen_name:
            existing_mention = mention
            break

    if not existing_mention:
        mentions_data[mentioned_user]["mentions"].append({
            "mentionBy": user_screen_name,
            "tweets": [tweet_id]
        })
    else:
        existing_mention["tweets"].append(tweet_id)

    mentions_data[mentioned_user]["receivedMentions"] += 1

def create_mention_json(tweets):
    mentions_data = {}

    for tweet in tweets:
        user_screen_name, retweeted, _, _ = process_user_tweet(tweet)

        if not retweeted and user_screen_name:
            mentioned_users = {mention["screen_name"] for mention in tweet.get("entities", {}).get("user_mentions", []) if mention["screen_name"] != "null"}
            
            for mentioned_user in mentioned_users:
                update_mention_json_data(mentions_data, user_screen_name, mentioned_user, tweet["id_str"])

    sorted_mentions = sorted(mentions_data.values(), key=lambda x: x['receivedMentions'], reverse=True)
    return {'mentions': sorted_mentions}


def save_output(data, output_path):
    try:
        if isinstance(data, nx.Graph):
            nx.write_gexf(data, output_path)
        elif isinstance(data, dict) and output_path.endswith(".json"):
            with open(output_path, "w") as file:
                json.dump(data, file, indent=4)
        else:
            print(f"Unsupported data type for output: {output_path}")
    except Exception as e:
        print(f"Error saving output to {output_path}: {e}")

def process_output(args, tweets):
    if args["generate_rt_graph"]:
        rt_graph = create_retweet_graph(tweets)
        save_output(rt_graph, "rtp.gexf")

    if args["generate_rt_json"]:
        retweet_json = create_retweet_json(tweets)
        save_output(retweet_json, "rtp.json")

    if args["generate_mention_graph"]:
        mention_graph = create_mention_graph(tweets)
        save_output(mention_graph, "mentionp.gexf")

    if args["generate_mention_json"]:
        mention_json = create_mention_json(tweets)
        save_output(mention_json, "mentionp.json")

    if args["generate_co_rt_graph"]:
        coretweet_graph = create_coretweet_graph(tweets)
        save_output(coretweet_graph, "corrtwp.gexf")

    if args["generate_co_rt_json"]:
        coretweet_json = create_coretweet_json(tweets)
        save_output(coretweet_json, "corrtwp.json")


def process_arguments(raw_args):

    start_date = datetime.strptime(raw_args["start_date"], "%d-%m-%y") if raw_args["start_date"] else datetime.strptime("01-01-00", "%d-%m-%y")  
    end_date = datetime.strptime(raw_args["end_date"], "%d-%m-%y") if raw_args["end_date"] else datetime.strptime("01-01-24", "%d-%m-%y")


    hashtags = set()
    if raw_args["hashtags"]:
        with open(raw_args["hashtags"], 'r') as file:
            hashtags = {line.strip().lower().lstrip("#") for line in file.readlines()}

    return {
        "directory": raw_args["directory"],
        "start_date": start_date,
        "end_date": end_date,
        "hashtags": hashtags,
        "generate_rt_graph": raw_args["grt"],
        "generate_rt_json": raw_args["jrt"],
        "generate_mention_graph": raw_args["gm"],
        "generate_mention_json": raw_args["jm"],
        "generate_co_rt_graph": raw_args["gcrt"],
        "generate_co_rt_json": raw_args["jcrt"]
    }
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Arguments for generador.py", add_help=False)
    parser.add_argument("-d", "--directory", default="input", help="Path to the data directory")
    parser.add_argument("-fi", "--start-date", help="Start date")
    parser.add_argument("-ff", "--end-date", help="End date")
    parser.add_argument("-h", "--hashtags", help="List of hashtags")
    parser.add_argument("-grt", action="store_true", help="Create retweet graph")
    parser.add_argument("-jrt", action="store_true", help="Create retweet JSON")
    parser.add_argument("-gm", action="store_true", help="Create mention graph")
    parser.add_argument("-jm", action="store_true", help="Create mention JSON")
    parser.add_argument("-gcrt", action="store_true", help="Create co-retweet graph")
    parser.add_argument("-jcrt", action="store_true", help="Create co-retweet JSON")
    args = parser.parse_args(argv)
    args = vars(args)
    if "directory" not in args:
       args["directory"] = "data"
    return args

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = time.time()
    
    
    raw_args = parse_args(sys.argv[1:])
    args = process_arguments(raw_args)
    

    # Process tweet files
    tweets = get_tweets(args["directory"], args["start_date"], args["end_date"], args["hashtags"])

    # Create and save graphs and JSONs
    if(rank ==0):
        process_output(args,tweets)
        end_time = time.time()
        print(end_time - start_time)

    # Delete temporary files and show execution time
    # ...
    # Calculate and display execution time
    

if __name__ == "__main__":
    main()

