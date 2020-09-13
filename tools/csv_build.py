import sys
import argparse
import pickle
import pandas as pd

def build_df(posts_all, class_label = 2):
    site = []
    url = []
    published = []
    site_type = []
    title = []
    text = []
    label = []
    for i in range(len(posts_all)):
        site.append(posts_all[i]['thread_info']['site'])
        url.append(posts_all[i]['thread_info']['url'])
        published.append(posts_all[i]['thread_info']['published'])
        site_type.append(posts_all[i]['thread_info']['site_type'])
        title.append(posts_all[i]['title'])
        text.append(posts_all[i]['text'])
        label.append(class_label)
    
    data = {'site': site, 'url': url, 'published': published, 'site_type': site_type, 'title': title, 'text': text, 'label': label}
    df = pd.DataFrame(data, columns = list(data.keys()))
    
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--center_articles_path", type=str, help="location of web scraped data for neutral news sources")
    parser.add_argument("--right_articles_path", type=str, help="location of web scraped data for right leaning news sources")
    parser.add_argument("--left_articles_path", type=int, help="location of web scraped data for left leaning news sources")
    parser.add_argument("--write_path", type=int, help="directory location to write concatenated dataset")
    args = parser.parse_args()

    posts_center = pickle.load(open(args.center_articles_path, "rb" ))
    posts_right = pickle.load(open(args.right_articles_path, "rb" ))
    posts_left = pickle.load(open(args.left_articles_path, "rb"))

    df_center = build_df(posts_center,1)
    df_right = build_df(posts_right,2)
    df_left = build_df(posts_left,0)

    df = pd.concat([df_center,df_left,df_right],ignore_index = True)
    df.to_csv(args.write_path, index=False)