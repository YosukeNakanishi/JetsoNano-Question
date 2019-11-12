# YosukeNakanihis

## 質問

お世話になります。

工場における外観検査支援のため、現在評価用にJetsonNano開発者キットを使用しています。

簡易の評価のためのプログラムをPyTorchにて作成、推論を実施しました。しかし推論に30秒ほどかかっており「推論時にGPUが使われていないであろうこと」を、以下に示す「確認したこと」から確認しました。

そこで質問なのですが、GPU/CUDAを使用する際に、JetsonNano本体側に設定しておく項目などはありますか？



## 試したこと

CPUで動作するプログラム、GPUで動作するプログラム2つのプログラムを用いて推論時間を実測。

```GPU版
# coding: utf-8

# パッケージのimport
import numpy as np
import json
from PIL import Image
import torch
import torchvision
from torchvision import models, transforms

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

use_pretrained = True  # 学習済みのパラメータを使用
net = models.vgg16(pretrained=use_pretrained)
net.eval()  # 推論モードに設定

# 入力画像の前処理のクラス
class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),  # 短い辺の長さがresizeの大きさになる
            transforms.CenterCrop(resize),  # 画像中央をresize × resizeで切り取り
            transforms.ToTensor(),  # Torchテンソルに変換
            transforms.Normalize(mean, std)  # 色情報の標準化
        ])

    def __call__(self, img):
        return self.base_transform(img)

# 画像前処理の動作を確認
# 1. 画像読み込み
image_file_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)  # [高さ][幅][色RGB]

# 3. 画像の前処理と処理済み画像の表示
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)  # torch.Size([3, 224, 224])

# 出力結果からラベルを予測する後処理クラスを作成
ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
ILSVRC_class_index

# 出力結果からラベルを予測する後処理クラス
class ILSVRCPredictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().cpu().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

# ネットワークをGPUへ
net = net.to(device)

# 学習済みVGGモデルで手元の画像を予測
# ILSVRCのラベル情報をロードし辞意書型変数を生成します
ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
predictor = ILSVRCPredictor(ILSVRC_class_index)

# 入力画像を読み込む
image_file_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)  # [高さ][幅][色RGB]

# 前処理の後、バッチサイズの次元を追加する
transform = BaseTransform(resize, mean, std)  # 前処理クラス作成
img_transformed = transform(img)  # torch.Size([3, 224, 224])
inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])

# GPUが使えるならGPUにデータを送る
inputs = inputs.to(device)

# モデルに入力し、モデル出力をラベルに変換する
out = net(inputs)  # torch.Size([1, 1000])
result = predictor.predict_max(out)

# 予測結果を出力する
print("入力画像の予測結果：", result)
```



## 確認したこと

#### プログラムのどこで時間が掛かっているかの確認

```
out = net(inputs)  # torch.Size([1, 1000])
```

にて、処理時間を要していることを実測で確認。



#### フレームワークが正しくインストールされているかの確認

```
jetson@jetson-desktop:~$ python3
Python 3.6.8 (default, Oct  7 2019, 12:59:55) 
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.

> > > import torch
> > > print(torch.__version__)
> > > 1.2.0a0+8554416
> > > print('CUDA available: ' + str(torch.cuda.is_available()))
> > > CUDA available: True
> > > a = torch.cuda.FloatTensor(2).zero_()
> > > print('Tensor a = ' + str(a))
> > > Tensor a = tensor([0., 0.], device='cuda:0')
> > > b = torch.randn(2).cuda()
> > > print('Tensor b = ' + str(b))
> > > Tensor b = tensor([0.4261, 2.1705], device='cuda:0')
> > > c = a + b
> > > print('Tensor c = ' + str(c))
> > > Tensor c = tensor([0.4261, 2.1705], device='cuda:0')

> > > import torchvision
> > > print(torchvision.__version__)
> > > 0.2.2

```

https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/



#### GPUの動作を確認

tegrastats と gpuGraph を使い確認したが、若干の動きはあるものの通常の画面操作と同程度なので、GPUが動いていないように見受けられる。



## 開発環境

・JetsonNano　jetson-nano-sd-r32.2-2019-07-16　でイメージを作成

・ubntu 18.04

・PyTorch v1.3.0



## 参考文献(サイト)

[PyTorchでBERTなど各種DLモデルを作りながら学ぶ書籍を執筆しました](https://qiita.com/sugulu/items/07253d12b1fc72e16aba)
[YutaroOgawa/pytorch_advanced](<https://github.com/YutaroOgawa/pytorch_advanced/tree/master/1_image_classification>)の[1-1_load_vgg.ipynb](https://github.com/YutaroOgawa/pytorch_advanced/blob/master/1_image_classification/1-1_load_vgg.ipynb)を参照
