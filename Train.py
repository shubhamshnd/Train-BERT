import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

# Define the custom dataset
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = str(self.data.iloc[index, 0])
        context = str(self.data.iloc[index, 1])
        answer = str(self.data.iloc[index, 2])

        encoding = self.tokenizer.encode_plus(
            question,
            context,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        start_idx = context.find(answer)
        end_idx = start_idx + len(answer)

        if start_idx == -1 or end_idx == -1:
            start_positions = torch.tensor(self.max_len - 1)
            end_positions = torch.tensor(self.max_len - 1)
        else:
            start_positions = encoding.char_to_token(0, start_idx)
            end_positions = encoding.char_to_token(0, end_idx - 1)

            if start_positions is None or end_positions is None:
                start_positions = torch.tensor(self.max_len - 1)
                end_positions = torch.tensor(self.max_len - 1)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': start_positions.clone().detach(),
            'end_positions': end_positions.clone().detach()
        }

# Load the data
df = pd.read_csv("/mnt/data/qa_train.csv")

# Initialize the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Create the dataset and dataloader
train_dataset = QADataset(df, tokenizer, max_len=512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Set up the optimizer and scheduler using PyTorch's AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_train_loss}")

# Save the model
model.save_pretrained("./qa_model")
tokenizer.save_pretrained("./qa_model")
