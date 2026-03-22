import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 8 # 训练难度相较于分类器更高，因此提高训练轮次
LEARNING_RATE = 2e-5
MODEL_NAME = 'C:/Users/86155/Desktop/PythonProject/model/bert-base-uncased'   # 英文数据

# 标签列表：只保留材质、颜色、品牌
LABEL_LIST = ['O',
              'B-material', 'I-material',
              'B-color', 'I-color',
              'B-brand', 'I-brand']
label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}

# 预定义颜色词库（用于 Multi-color 匹配）
COLOR_WORDS = {'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'grey', 'gray',
               'orange', 'purple', 'pink', 'cyan', 'magenta', 'navy', 'beige', 'maroon',
               'turquoise', 'violet', 'indigo', 'gold', 'silver', 'bronze', 'multicolor',
               'multi-color', 'multi colour', 'colourful', 'colorful'}

# ==================== 数据加载与预处理 ====================
def parse_parameters(param_str):
    """
    从 parameters 字符串中解析出材质、颜色、品牌三个字段。
    缺失字段填充空字符串。
    """
    parts = param_str.split(', ')
    # 正常情况有5个字段：category, sub_category, material, color, brand
    if len(parts) >= 5:
        return {
            'material': parts[2],
            'color': parts[3],
            'brand': parts[4]
        }
    else:
        # 异常情况：尽力解析，缺失字段留空
        result = {'material': '', 'color': '', 'brand': ''}
        fields_order = ['material', 'color', 'brand']
        for i, field in enumerate(fields_order):
            # 从第2个字段开始（索引2对应material）
            idx = i + 2
            if idx < len(parts):
                result[field] = parts[idx]
        return result

def find_color_words_in_text(text, color_set):
    """返回文本中包含的颜色词列表（小写）"""
    text_lower = text.lower()
    found = []
    for color in color_set:
        if color in text_lower:
            found.append(color)
    return found

def tokenize_and_align_labels(text, entities, tokenizer, max_len):
    """
    对文本进行 tokenize，并根据改进的实体匹配规则生成 BIO 标签序列。
    text: 原始文本（标题+描述）
    entities: 字典，包含 'material', 'color', 'brand'
    返回: input_ids, attention_mask, labels
    """
    entity_spans = []

    # 1. 品牌匹配（精确/子串）
    brand_val = entities.get('brand', '')
    if brand_val and not pd.isna(brand_val):
        brand_lower = brand_val.lower()
        start = 0
        while True:
            pos = text.lower().find(brand_lower, start)
            if pos == -1:
                break
            end = pos + len(brand_val)
            entity_spans.append(('brand', pos, end))
            start = end

    # 2. 颜色匹配（支持多值和 Multi-color 特殊处理）
    color_val = entities.get('color', '')
    if color_val and not pd.isna(color_val):
        if color_val.lower() in ['multicolor', 'multi-color', 'multi colour']:
            # 特殊处理：在文本中查找多个颜色词
            found_colors = find_color_words_in_text(text, COLOR_WORDS)
            if len(found_colors) >= 2:
                # 标记整个文本区间（从第一个颜色开始到最后一个颜色结束）
                # 更精确的做法是分别标记每个颜色词，这里简化：将整个文本视为一个实体
                # 为了简单，我们直接为每个出现的颜色词单独生成实体
                for col in found_colors:
                    start = 0
                    while True:
                        pos = text.lower().find(col, start)
                        if pos == -1:
                            break
                        end = pos + len(col)
                        entity_spans.append(('color', pos, end))
                        start = end
            else:
                # 查找 "multicolor" 等词本身
                variants = ['multicolor', 'multi-color', 'multi colour', 'colorful']
                for variant in variants:
                    start = 0
                    while True:
                        pos = text.lower().find(variant, start)
                        if pos == -1:
                            break
                        end = pos + len(variant)
                        entity_spans.append(('color', pos, end))
                        start = end
        else:
            # 常规颜色值（可能包含逗号）
            color_parts = [c.strip() for c in color_val.split(',')] if ',' in color_val else [color_val]
            for col in color_parts:
                if not col:
                    continue
                col_lower = col.lower()
                start = 0
                while True:
                    pos = text.lower().find(col_lower, start)
                    if pos == -1:
                        break
                    end = pos + len(col)
                    entity_spans.append(('color', pos, end))
                    start = end

    # 3. 材质匹配（包含关系：参数值中的任意单词出现在文本中）
    material_val = entities.get('material', '')
    if material_val and not pd.isna(material_val):
        # 将材质值拆分为单词（按空格或连字符）
        words = set()
        # 去掉括号内容等（如果有）
        clean_mat = material_val.replace('(', ' ').replace(')', ' ').replace('-', ' ')
        for w in clean_mat.split():
            w_clean = w.strip().lower()
            if w_clean:
                words.add(w_clean)
        # 如果拆分后为空，则使用原值
        if not words:
            words.add(material_val.lower())
        for word in words:
            # 在文本中查找单词（作为子串）
            start = 0
            while True:
                pos = text.lower().find(word, start)
                if pos == -1:
                    break
                end = pos + len(word)
                entity_spans.append(('material', pos, end))
                start = end

    # 按起始位置排序
    entity_spans.sort(key=lambda x: x[1])

    # 使用 tokenizer 编码文本
    encoding = tokenizer(text,
                         truncation=True,
                         padding='max_length',
                         max_length=max_len,
                         return_offsets_mapping=True)
    offset_mapping = encoding['offset_mapping']

    # 初始化标签为 O
    labels = [label2id['O']] * max_len

    # 为每个 token 分配标签
    for field, start_char, end_char in entity_spans:
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end == 0:
                continue  # 跳过特殊 token
            if token_start >= start_char and token_end <= end_char:
                # 如果是实体的第一个 token
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
    input_ids_list = []
    attention_masks_list = []
    labels_list = []

    for _, row in data_df.iterrows():
        text = f"{row['title']}。{row['description']}"
        entities = parse_parameters(row['parameters'])
        encoded = tokenize_and_align_labels(text, entities, tokenizer, max_len)
        input_ids_list.append(encoded['input_ids'])
        attention_masks_list.append(encoded['attention_mask'])  # 注意键名
        labels_list.append(encoded['labels'])

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
                'attention_mask': self.attention_masks[idx],  # 保持与 DataLoader 中的键名一致
                'labels': self.labels[idx]
            }

    return ProductDataset(input_ids_list, attention_masks_list, labels_list)

# ==================== 模型定义与训练 ====================
def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    model.to(device)
    # 定义最佳 F1 分数和等待次数
    best_f1 = 0
    patience = 2
    wait = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")
        for batch in progress_bar:
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

            # 更新进度条显示 loss
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Train loss: {avg_loss:.4f}")

        # 验证
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                for i in range(len(labels)):
                    pred_seq = []
                    true_seq = []
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:  # 忽略填充
                            pred_seq.append(id2label[predictions[i][j].item()])
                            true_seq.append(id2label[labels[i][j].item()])
                    preds.append(pred_seq)
                    trues.append(true_seq)

        # 计算每个实体的 F1
        report = classification_report(trues, preds, digits=4, output_dict=False)
        print(report)

        f1 = f1_score(trues, preds)

        # 早停机制
        if f1 > best_f1:
            best_f1 = f1
            wait = 0
            torch.save(model.state_dict(), '../model/bert_ner/best_ner_model.pth')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


# ==================== 预测函数 ====================
def predict(text, model, tokenizer, device, max_len=MAX_LEN):
    """对单条文本进行预测，返回材质、颜色、品牌三个字段的字典"""
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

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [id2label[pred.item()] for pred in predictions]

    entities = {'material': [], 'color': [], 'brand': []}
    current_entity = None
    current_field = None
    for token, label in zip(tokens, labels):
        if label == 'O':
            if current_entity:
                entities[current_field].append(''.join(current_entity).replace('##', ''))
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

    if current_entity:
        entities[current_field].append(''.join(current_entity).replace('##', ''))

    result = {}
    for field, values in entities.items():
        result[field] = ', '.join(values) if values else ''
    return result

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 读取预处理后的数据
    df = pd.read_csv('C:/Users\86155\Desktop\PythonProject\data\processed\processed.csv')  # 确保文件路径正确
    print(f"数据总量: {len(df)}")

    # 2. 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}")

    # 3. 初始化 tokenizer 和模型
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
    model.load_state_dict(torch.load('../model/bert_ner/best_ner_model.pth'))
    model.to(device)

    # 7. 示例预测
    sample_text = "Solid Men Multicolor Track Pants。Yorker trackpants made from 100% rich combed cotton giving it a rich look."
    result = predict(sample_text, model, tokenizer, device)
    print("\n预测结果:")
    for k, v in result.items():
        print(f"{k}: {v}")