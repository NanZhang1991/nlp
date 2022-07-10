import torch
from transformers import BertTokenizer
from longformer.longformer import LongformerConfig, LongformerForSequenceClassification
from longformer.sliding_chunks import pad_to_window_size

model_path = "../../models/longformer_zh"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained("../../models/longformer_zh")
config = LongformerConfig.from_pretrained("../../models/longformer_zh")
config = LongformerConfig.from_json_file("../../models/longformer_zh/config.json")
config.problem_type = "multi_label_classification"
config.num_labels = 6
model = LongformerForSequenceClassification.from_pretrained("../../models/longformer_zh", config=config)
model.to(device)



import pandas as pd 

train_path = "../data/MultilabelSequenceClassification/toxic-comment-classification/train.csv.zip"
test_path = "../data/MultilabelSequenceClassification/toxic-comment-classification/test.csv.zip"
df = pd.read_csv(train_path)
df['label'] = df[df.columns[2:]].values.tolist()
new_df = df[['comment_text', 'label']].copy()
print(new_df.head())

train_size=0.9
test_data = pd.read_csv(test_path)[:100]
train_data = new_df.sample(frac=train_size,random_state=200).reset_index(drop=True)[:1000]
val_data = new_df.drop(train_data.index).reset_index(drop=True)[:100]

print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_data.shape))
print("TEST Dataset: {}".format(test_data.shape))


import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['comment_text']
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        inputs_single = {}
        inputs_single['input_ids'] = inputs['input_ids'][0]
        inputs_single['attention_mask'] = inputs['attention_mask'][0]
        inputs_single['token_type_ids'] = inputs['token_type_ids'][0]
        targets = self.targets[index]

        return inputs_single, targets


train_datset = CustomDataset(train_data, tokenizer, config.max_position_embeddings)
val_set = CustomDataset(val_data, tokenizer, config.max_position_embeddings)
train_loader = DataLoader(train_datset, batch_size=4, shuffle=True)
val_loader = DataLoader(train_datset, batch_size=4)


optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-05)
loss_fn = torch.nn.BCEWithLogitsLoss()
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(**inputs)
        loss = outputs.loss
        # Compute the loss and its gradients
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        print(1)
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            print(1)
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

epochs = 1
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
for epoch in range(1, epochs+1):
    print(f'epoch {epoch}:')  # 输出轮次

    avg_loss = train_one_epoch(epoch, writer)  # 获得一轮结束后平均损失
    
    # model.train(False)
    with torch.no_grad():  # 不求梯度
        running_vloss = 0.0
        for i, (vinputs, vlabels) in enumerate(val_loader):
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(**vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                        epoch)  # 写入训练集和测试集损失
                       
    if avg_loss < best_vloss:  # 如果平均损失小于最小的损失
        best_vloss = avg_loss  # 更新最小损失为平均损失
        model_path = f'../../models/longformer_test'
        torch.save(model, model_path)  # 将模型保存到该路径