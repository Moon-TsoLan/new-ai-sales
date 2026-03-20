import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
MAX_LEN = 128  # 序列最大长度（可根据实际情况调整）
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = 'C:/Users/86155/Desktop/PythonProject/model/bert-base-uncased'  # 已下载至本地 英文数据使用 uncased


# ==================== 数据准备 ====================
def load_and_prepare_data(csv_path):
    """读取CSV，从parameters中解析category和sub_category，构造文本和标签"""
    df = pd.read_csv(csv_path)
    # 解析 parameters 列，格式如 "Clothing and Accessories, Bottomwear, Cotton Blend, Multicolor, York"
    def parse_params(param_str):
        parts = param_str.split(', ')
        # 返回前两个字段作为主类和次类，若字段不足则填充'unknown'
        category = parts[0] if len(parts) > 0 else 'unknown'
        sub_category = parts[1] if len(parts) > 1 else 'unknown'
        return category, sub_category

    # 应用解析
    df[['category', 'sub_category']] = df['parameters'].apply(
        lambda x: pd.Series(parse_params(str(x)))
    )
    # 组合文本：title + description（中间加空格）
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    # 处理缺失标签：若category或sub_category为空，填充为'unknown'
    df['category'] = df['category'].fillna('unknown')
    df['sub_category'] = df['sub_category'].fillna('unknown')
    return df


def encode_labels(df, category_encoder=None, sub_encoder=None):
    """对category和sub_category进行标签编码"""
    if category_encoder is None:
        category_encoder = LabelEncoder()
        df['category_id'] = category_encoder.fit_transform(df['category'])
    else:
        df['category_id'] = category_encoder.transform(df['category'])

    if sub_encoder is None:
        sub_encoder = LabelEncoder()
        df['sub_category_id'] = sub_encoder.fit_transform(df['sub_category'])
    else:
        df['sub_category_id'] = sub_encoder.transform(df['sub_category'])

    return category_encoder, sub_encoder


# ==================== 自定义 Dataset ====================
class ProductDataset(Dataset):
    def __init__(self, texts, category_ids, sub_ids, tokenizer, max_len):
        self.texts = texts
        self.category_ids = category_ids
        self.sub_ids = sub_ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category_label': torch.tensor(self.category_ids[idx], dtype=torch.long),
            'sub_label': torch.tensor(self.sub_ids[idx], dtype=torch.long)
        }


# ==================== 多输出分类模型 ====================
class MultiOutputBertClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_categories, num_sub_categories):
        super().__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.1)
        self.category_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_categories)
        self.sub_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_sub_categories)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] 向量进行分类
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        sub_logits = self.sub_classifier(pooled_output)
        return category_logits, sub_logits


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cat_labels = batch['category_label'].to(device)
            sub_labels = batch['sub_label'].to(device)

            optimizer.zero_grad()
            cat_logits, sub_logits = model(input_ids, attention_mask)
            loss_cat = loss_fn(cat_logits, cat_labels)
            loss_sub = loss_fn(sub_logits, sub_labels)
            loss = loss_cat + loss_sub
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # 可选：每 50 步显示一次当前 loss
            if batch_idx % 50 == 0:
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Train loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0
        all_cat_preds, all_cat_trues = [], []
        all_sub_preds, all_sub_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                cat_labels = batch['category_label'].to(device)
                sub_labels = batch['sub_label'].to(device)

                cat_logits, sub_logits = model(input_ids, attention_mask)
                loss_cat = loss_fn(cat_logits, cat_labels)
                loss_sub = loss_fn(sub_logits, sub_labels)
                val_loss += (loss_cat + loss_sub).item()

                cat_preds = torch.argmax(cat_logits, dim=1)
                sub_preds = torch.argmax(sub_logits, dim=1)
                all_cat_preds.extend(cat_preds.cpu().numpy())
                all_cat_trues.extend(cat_labels.cpu().numpy())
                all_sub_preds.extend(sub_preds.cpu().numpy())
                all_sub_trues.extend(sub_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        cat_acc = accuracy_score(all_cat_trues, all_cat_preds)
        sub_acc = accuracy_score(all_sub_trues, all_sub_preds)
        print(f"Val loss: {avg_val_loss:.4f}, Category Acc: {cat_acc:.4f}, Sub Acc: {sub_acc:.4f}")

        # 保存最佳模型（根据验证总损失）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_classifier.pth')
            print("Best model saved.")

    # 加载最佳模型
    model.load_state_dict(torch.load('best_classifier.pth'))
    return model


# ==================== 预测函数 ====================
def predict(text, model, tokenizer, device, category_encoder, sub_encoder, max_len=MAX_LEN):
    """对单条文本预测主类和次类，返回原始标签"""
    model.eval()
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        cat_logits, sub_logits = model(input_ids, attention_mask)
        cat_pred_id = torch.argmax(cat_logits, dim=1).item()
        sub_pred_id = torch.argmax(sub_logits, dim=1).item()

    category = category_encoder.inverse_transform([cat_pred_id])[0]
    sub_category = sub_encoder.inverse_transform([sub_pred_id])[0]
    return category, sub_category


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 加载数据
    # 注意：这里假设你的预处理后的CSV文件名为 'processed.csv'，且包含 'title', 'description', 'category', 'sub_category' 列
    # 如果你的文件列名不同，请相应修改
    df = load_and_prepare_data('C:/Users/86155/Desktop/PythonProject/data/processed/processed.csv')
    print(f"数据总量: {len(df)}")

    # 2. 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}")

    # 3. 标签编码（使用训练集fit，验证集transform）
    category_encoder, sub_encoder = encode_labels(train_df)
    # 对验证集编码
    _, _ = encode_labels(val_df, category_encoder, sub_encoder)

    # 4. 初始化 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # 临时加载，后面会替换
    # 注意：这里我们先加载了 BertForSequenceClassification，但实际我们自定义模型，所以只取其 BERT 部分
    bert_model = bert_model.bert  # 提取 BERT 编码器

    num_categories = len(category_encoder.classes_)
    num_sub_categories = len(sub_encoder.classes_)
    model = MultiOutputBertClassifier(bert_model, num_categories, num_sub_categories)

    # 5. 创建数据集
    train_dataset = ProductDataset(train_df['text'].values,
                                   train_df['category_id'].values,
                                   train_df['sub_category_id'].values,
                                   tokenizer, MAX_LEN)
    val_dataset = ProductDataset(val_df['text'].values,
                                 val_df['category_id'].values,
                                 val_df['sub_category_id'].values,
                                 tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    trained_model = train_model(model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE)

    # 7. 测试示例
    sample_text = "Solid Men Multicolor Track Pants. Yorker trackpants made from 100% rich combed cotton."
    cat, sub = predict(sample_text, trained_model, tokenizer, device, category_encoder, sub_encoder)
    print(f"\n预测结果: 主类 = {cat}, 次类 = {sub}")

    # 可选：保存编码器和模型配置，供后续部署使用
    import joblib

    joblib.dump(category_encoder, 'category_encoder.pkl')
    joblib.dump(sub_encoder, 'sub_encoder.pkl')
    torch.save(trained_model.state_dict(), 'best_classifier.pth')
    print("模型和编码器已保存。")