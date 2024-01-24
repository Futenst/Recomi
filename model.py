from transformers import MLukeTokenizer, LukeModel
import sentencepiece as spm
import torch
import scipy.spatial
import pandas as pd
import numpy as np


class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx : batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch, padding="longest", truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            ).to("cpu")

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


# 既存モデルの読み込み
def recommend(query):

        MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
        model = SentenceLukeJapanese(MODEL_NAME)

        # 説明文を入れるリストを作成
        sentences = []

        # CSVファイルのパスを指定
        csv_file_path = 'Comics_df - Comics_df.csv'

        # 読み込む列の名前を指定
        target_column_name = ['info']

        # CSVファイルをDataFrameとして読み込む
        data = pd.read_csv(csv_file_path)

        print(data)
        
        # 指定した列のデータをリストに追加
        #sentences = np.array(data['info'].tolist())
        sentences = data[target_column_name].tolist()

        # 標準入力で、理想のビールのイメージを文章で受け取る
        #sentences = np.append(sentences,query)
        sentences.append(query)
        query_embedding_vector = model.encode([query],batch_size=8)

        # カクテルの説明文、受け取った文章をエンコード（ベクトル表現に変換）
        #sentence_embeddings = model.encode(sentences, batch_size=32)
        #sentence_embeddings_np = sentence_embeddings.numpy()

        #sentence_embeddings = model.encode(sentences, batch_size=8)
        #print(sentence_embeddings.shape)
        #torch.save(sentence_embeddings,"sentence_embeddings.pt")

        sentence_embeddings = torch.vstack((sentence_embeddings, query_embedding_vector))
        print(sentence_embeddings.shape)

        # 類似度上位1つを出力
        closest_n = 1 #5にすると上位一位のタイトルが繰り返されてしまう。

        distances = scipy.spatial.distance.cdist(
            [sentence_embeddings[-1]], sentence_embeddings, metric="cosine"
        )[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nオススメの漫画は:")
        #index= data[data['volumes','info']]==sentences[results[1][0]].strip()].index[0] 
        index = data[data['info'] == sentences[results[1][0]].strip()].index[0]

        return data.iloc[index,1],data.iloc[index,5],data.iloc[index,7],data.iloc[index,8],data.iloc[index,2]