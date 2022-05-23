import torch 
from tqdm import tqdm 

def get_product_ratings(df_test, model, reviewer2int, product2int, int2product):
    df_test_1 = df_test.groupby('reviewerID').filter(lambda x: len(x) > 10)
    reviewers_in_test_dataset = df_test_1['reviewerID'].unique()
    products_in_test_dataset = df_test_1['asin'].unique()
    
    top10products_user = {}
    
    for reviewer in tqdm(reviewers_in_test_dataset):
        reviewer_id = reviewer2int[reviewer]
        user2product_ratings = []
        for product in products_in_test_dataset:
            product_id = product2int[product]
            product_rating = model(torch.tensor(product_id, dtype=torch.long).unsqueeze(0), torch.tensor(reviewer_id, dtype=torch.long).unsqueeze(0))
            user2product_ratings.append(product_rating)
        top10products = torch.topk(torch.tensor(user2product_ratings), 10).indices
        top10products_user[reviewer] = [int2product[i.item()] for i in top10products]
    return top10products_user