import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3


class ContentBasedRecommender:

    # def second_db(self, product):
    #     db_connect = sqlite3.connect(r'F:\pycharm\second_db.sqlite3')
    #     cr = db_connect.cursor()
    #     cr.execute('INSERT INTO home_product VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)', (product['product_id'], product['Name'],
    #                 product['description'], product['Production_country'], product['image'], product['created_at'],
    #                 product['user_id'], product['category_id'], product['price']))
    #     db_connect.commit()
    #     db_connect.close()


    def prepare(self):
        from .models import Product

        df = pd.DataFrame(list(Product.objects.all().values('product_id', 'Name', 'description', 'category__Name')))

        df.rename(columns={'product_id': 'id', 'Name': 'Machine Name', 'category__Name': 'Category'}, inplace=True)

        df['Machine Name'] = df['Machine Name'].apply(lambda x: x.split())
        df['Category'] = df['Category'].apply(lambda x: x.split())
        df['description'] = df['description'].apply(lambda x: x.split())
        df['combination_column'] = df.description + df.Category + df['Machine Name']
        final_df = df[['id', 'Machine Name', 'combination_column']]
        final_df['Machine Name'] = final_df['Machine Name'].apply(lambda x: ' '.join(x))
        final_df['combination_column'] = final_df['combination_column'].apply(lambda x: ' '.join(x))
        return final_df

    def get_similarity(self):

        transformer = TfidfVectorizer(stop_words='english', max_features=5000)
        final_df = self.prepare()
        vector = transformer.fit_transform(final_df['combination_column'])
        vectors = vector.toarray()
        similarity = cosine_similarity(vectors, vectors)
        return similarity, final_df

    def get_recommendation(self, machine_id):
        similarity, final_df = self.get_similarity()
        if machine_id in final_df['id'].unique():
            return 0
        machine_index = final_df[final_df['id'] == machine_id].index[0]
        distances = similarity[machine_index]
        machines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        reco_list = []
        for i in machines_list:
            reco_list.append(final_df.iloc[i[0]]['id'])
        return reco_list
