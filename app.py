import os
from flask import (
     Flask, 
     request, 
     render_template)
from model import recommend #model.pyからrecommend関数をインポート
import torch
import math
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) #トップページのルーティング
def top():
    return render_template('top.html')

@app.route('/kibun1', methods=['GET'])   #気分入力ページのルーティング
def kibun1():
    return render_template('kibun1.html')

@app.route('/kibun2', methods=['GET','POST'])   #出力ページのルーティング
def kibun2():
    if request.method == "GET":
        return render_template('kibun2.html')
    elif request.method == "POST":
        favs = request.form.getlist("fav")#name属性がfavのcheckboxから複数の値を取得
        data_str = ",".join(favs)
        #name,volumes,imageurl,url,author = recommend(data_str)

        sentence_embeddings = torch.load("sentence_embeddings.pt") #ベクトルデータ読み込み
        try:

            fav = request.form.get("fav")# name属性がfavのtextボックスから単一の値を取得
           
            title, author, theme, genre, volumes, info, img_element, paga_url = recommend(fav,sentence_embeddings)
            #if math.isnan(apple_url):
            #     apple_url = None
            #if isinstance(apple_url, float) and math.isnan(apple_url):
                #apple_url = None
            #if isinstance(google_url, float) and math.isnan(google_url):
                #google_url = None
            # isinstance(web_url, float) and math.isnan(web_url):
            theme = request.form.getlist("theme") #name属性がfavのtextボックスから単一の値を取得
            # results = [game['title'] for game in data if game['Duration'] == fav]            
            results=[]

            
            def search_data_by_theme(csv_filename, target_theme):
                results = []

                csv_file_path = 'Comics_df - Comics_df.csv'
                data = pd.read_csv(csv_file_path)

                with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                    reader = data.DictReader(csvfile)
                    
                    #for row in reader:
                    for index, row in data.iterrows():
                        if row['theme'] == target_theme:
                            result_dict = {key: row[key] for key in ['title','author','theme','genre','volumes','img_element','page_url']}
                            results.append(result_dict)

                return results

            
            results = search_data_by_theme("Comics_df - Comics_df.csv", theme[0])
                #web_url = None
            return render_template('kibun2.html', name=title,volumes=volumes,imageurl=img_element,url=paga_url,author=author)#左辺がHTML、右辺がPython側の変数
        except KeyError as e:
            return f"KeyError: {e}"

if __name__ == "__main__":
    app.run(debug=True)