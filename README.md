# CLARE Reproduction

CLARE データセットを用いて、認知負荷推定を再現実装するためのリポジトリです。  
ECG、EDA、EEG、Gaze のマルチモーダル信号を読み込み、前処理、特徴抽出、学習、評価までを一通り試せる構成になっています。

このリポジトリでは、以下の 2 系統のアプローチを扱っています。

- 古典的機械学習モデル
- 深層学習モデル（CNN / Transformer）

## 概要

`src/` 以下に、データ読み込み、前処理、特徴量生成、モデル定義、評価処理を分割して配置しています。  
データは 10 秒単位のセグメントに分割され、ラベルは `5` 以上を高認知負荷、`5` 未満を低認知負荷として 2 値化しています。

評価は主に次の 2 パターンに対応しています。

- 10-fold Cross Validation
- LOSO（Leave-One-Subject-Out）

## データセット

対象データセットは CLARE です。

- Borealis Data: [https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT)

`download_data.sh` は自動ダウンロード用というより、取得先と配置方法の案内用スクリプトです。  
データ取得後は、リポジトリ直下の `data/` に以下のような構成で展開してください。

```text
data/
├── ECG/
├── EDA/
├── EEG/
├── Gaze/
└── Labels/
```

各モダリティ配下には `P01` 〜 の参加者ディレクトリ、その中に各セッションの CSV が入る想定です。

## セットアップ

### 1. 依存関係のインストール

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. データ配置

```bash
bash download_data.sh
```

表示される案内に従って CLARE データセットを取得し、`./data` に展開してください。

## 実行方法

### 古典的機械学習モデル

```bash
python train_classical.py --data_root ./data --modalities ECG EDA EEG Gaze
```

特徴量抽出を行ったうえで、複数の古典的モデルを評価します。

### 深層学習モデル

CNN の実行例:

```bash
python train_dl.py --data_root ./data --model cnn
```

Transformer の実行例:

```bash
python train_dl.py --data_root ./data --model transformer
```

評価方法や学習条件はオプションで切り替えられます。

```bash
python train_dl.py \
  --data_root ./data \
  --model cnn \
  --scheme both \
  --epochs 100 \
  --batch_size 256 \
  --modalities ECG EDA EEG Gaze
```

主なオプション:

- `--model`: `cnn` または `transformer`
- `--scheme`: `kfold` / `loso` / `both`
- `--epochs`: 学習エポック数
- `--batch_size`: バッチサイズ
- `--modalities`: 使用するモダリティの一覧

## ディレクトリ構成

```text
.
├── CLARE_final.ipynb
├── CLARE_reproduction.ipynb
├── download_data.sh
├── requirements.txt
├── train_classical.py
├── train_dl.py
└── src/
    ├── data_loader.py
    ├── evaluation/
    ├── features/
    ├── models/
    └── preprocessing/
```

### 補足

- `CLARE_reproduction.ipynb` / `CLARE_final.ipynb`  
  実験や検証をノートブックで確認するためのファイルです。
- `src/preprocessing/`  
  各モダリティの前処理をまとめています。
- `src/features/`  
  古典的機械学習向けの特徴量抽出処理をまとめています。
- `src/models/`  
  古典モデル、CNN、Transformer の定義を含みます。
- `src/evaluation/`  
  評価指標やクロスバリデーション処理を含みます。

## 注意点

- `data/` はリポジトリに含めていません。
- 実行にはデータセットの手動取得が必要です。
- 深層学習部分は、環境によって実行時間が長くなることがあります。

## 今後の改善候補

- 実験結果の整理と README への追記
- 推奨 Python バージョンの明記
- 学習済みモデル保存機能の追加
- 実験設定ファイル化
