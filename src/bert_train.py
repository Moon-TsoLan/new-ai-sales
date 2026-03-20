import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'   # 中文BERT，如果数据是英文可选'bert-base-uncased'
LABEL_LIST = ['O',
              'B-category', 'I-category',
              'B-sub_category', 'I-sub_category',
              'B-material', 'I-material',
              'B-color', 'I-color',
              'B-brand', 'I-brand']
label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}

# ==================== 数据加载与预处理 ====================
def parse_parameters(param_str):
    """
    将parameters字符串拆分为五个字段，返回字典。
    如果某个字段缺失，则对应值为空字符串（''）。
    """
    parts = param_str.split(', ')
    # 正常情况有5个字段
    if len(parts) == 5:
        return {
            'category': parts[0],
            'sub_category': parts[1],
            'material': parts[2],
            'color': parts[3],
            'brand': parts[4]
        }
    # 异常情况：字段数量不足或过多，我们仍尽力解析，缺失的字段填充空字符串
    else:
        fields = ['category', 'sub_category', 'material', 'color', 'brand']
        result = {field: '' for field in fields}
        for i, field in enumerate(fields):
            if i < len(parts):
                result[field] = parts[i]
        return result

def tokenize_and_align_labels(text, entities, tokenizer, max_len):
    """
    对文本进行tokenize，并根据实体位置生成BIO标签序列
    text: 原始文本（标题+描述）
    entities: 字典，key为字段名，value为字段值（字符串）
    返回: input_ids, attention_mask, labels
    """
    # 将字段值转换为列表（处理多值，如颜色可能包含逗号）
    entity_spans = []
    for field, value in entities.items():
        if not value or pd.isna(value):
            continue
        # 如果值包含逗号，则拆分为多个值
        values = [v.strip() for v in value.split(',')] if ',' in value else [value]
        for v in values:
            # 在文本中查找所有出现的位置（不区分大小写）
            start = 0
            while True:
                pos = text.lower().find(v.lower(), start)
                if pos == -1:
                    break
                end = pos + len(v)
                entity_spans.append((field, pos, end))
                start = end  # 继续向后查找，避免重叠

    # 按起始位置排序
    entity_spans.sort(key=lambda x: x[1])

    # 使用tokenizer编码文本
    encoding = tokenizer(text,
                         truncation=True,
                         padding='max_length',
                         max_length=max_len,
                         return_offsets_mapping=True)  # 获取字符偏移
    offset_mapping = encoding['offset_mapping']

    # 初始化标签为O
    labels = [label2id['O']] * max_len

    # 为每个token分配标签
    for field, start_char, end_char in entity_spans:
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            # 忽略特殊token的偏移量（[CLS], [SEP], [PAD]）
            if token_start == token_end == 0:
                continue
            # 判断token是否在实体范围内
            if token_start >= start_char and token_end <= end_char:
                # 如果是实体第一个token
                if token_start == start_char or labels[idx] == label2id['O']:
                    labels[idx] = label2id[f'B-{field}']
                else:
                    labels[idx] = label2id[f'I-{field}']

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': labels
    }

def create_dataset(data_df, tokenizer, max_len):
    """将DataFrame转换为PyTorch Dataset"""
    texts = []
    labels = []
    for _, row in data_df.iterrows():
        # 组合标题和描述，中间用句号分隔
        text = f"{row['title']}。{row['description']}"
        # 解析参数
        entities = parse_parameters(row['parameters'])
        # 生成标签
        encoded = tokenize_and_align_labels(text, entities, tokenizer, max_len)
        texts.append(encoded['input_ids'])
        labels.append(encoded['labels'])

    class ProductDataset(Dataset):
        def __init__(self, input_ids, attention_masks, labels):
            self.input_ids = torch.tensor(input_ids, dtype=torch.long)
            self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'labels': self.labels[idx]
            }

    # 提取input_ids和attention_mask
    input_ids = [x['input_ids'] for x in texts]
    attention_masks = [x['attention_mask'] for x in texts]
    labels = [x['labels'] for x in labels]

    return ProductDataset(input_ids, attention_masks, labels)

# ==================== 模型定义与训练 ====================
def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    model.to(device)
    best_f1 = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Train loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # 转为列表，忽略padding位置
                for i in range(len(labels)):
                    pred_seq = []
                    true_seq = []
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:  # 忽略填充
                            pred_seq.append(id2label[predictions[i][j].item()])
                            true_seq.append(id2label[labels[i][j].item()])
                    preds.append(pred_seq)
                    trues.append(true_seq)

        # 计算准确率、召回率、F1
        report = classification_report(trues, preds, digits=4)
        print(report)
        # 计算总体F1（micro）
        from seqeval.metrics import f1_score
        f1 = f1_score(trues, preds)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with F1: {f1:.4f}")

# ==================== 预测函数 ====================
def predict(text, model, tokenizer, device, max_len=MAX_LEN):
    """对单条文本进行预测，返回五个字段的字典"""
    model.eval()
    encoding = tokenizer(text,
                         truncation=True,
                         padding='max_length',
                         max_length=max_len,
                         return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0]

    # 将预测的token标签转换为实体
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [id2label[pred.item()] for pred in predictions]
    # 忽略特殊token
    entities = {field: [] for field in ['category', 'sub_category', 'material', 'color', 'brand']}
    current_entity = None
    current_field = None
    for token, label in zip(tokens, labels):
        if label == 'O':
            if current_entity:
                # 保存当前实体
                field = current_field
                entities[field].append(''.join(current_entity).replace('##', ''))
                current_entity = None
                current_field = None
            continue
        if label.startswith('B-'):
            if current_entity:
                entities[current_field].append(''.join(current_entity).replace('##', ''))
            current_field = label[2:]
            current_entity = [token]
        elif label.startswith('I-'):
            if current_entity and label[2:] == current_field:
                current_entity.append(token)
            else:
                # 异常情况，忽略
                pass

    if current_entity:
        entities[current_field].append(''.join(current_entity).replace('##', ''))

    # 合并多值实体
    result = {}
    for field, values in entities.items():
        result[field] = ', '.join(values) if values else ''
    return result

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 读取预处理后的数据
    df = pd.read_csv('processed.csv')  # 请替换为实际文件路径
    print(f"数据总量: {len(df)}")

    # 2. 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}")

    # 3. 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_LIST))

    # 4. 创建数据集
    train_dataset = create_dataset(train_df, tokenizer, MAX_LEN)
    val_dataset = create_dataset(val_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    train_model(model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE)

    # 6. 加载最佳模型并测试
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)

    # 7. 示例预测
    sample_text = "Solid Men Multicolor Track Pants。Yorker trackpants made from 100% rich combed cotton giving it a rich look."
    result = predict(sample_text, model, tokenizer, device)
    print("\n预测结果:")
    for k, v in result.items():
        print(f"{k}: {v}")