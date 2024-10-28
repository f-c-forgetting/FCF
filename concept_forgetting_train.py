import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from transformers import CLIPTokenizer, CLIPTextModel
import argparse 

class Config:
    def __init__(self):
        self.version = "CompVis/stable-diffusion-v1-4"
        self.device = torch.device("cuda")
        self.max_length = 77
        self.n_repeat = 1
        self.normalize = True

        self.scheduler_step_size = 3
        self.scheduler_gamma = 0.1
        self.learning_rate = 0.000025
        self.num_epochs = 60
        self.save_interval = 60
        self.eta = 0.25

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument(
        "--input_prompts", 
        type=str, 
        required=True, 
        help="Path to the input file containing the prompts"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True, 
        help="Path to save the trained model"
    )

    args = parser.parse_args()

    if not args.save_path.endswith('.pt'):
        args.save_path += '.pt'

    return args

def main():
    args = parse_args()

    config = Config()

    tokenizer = CLIPTokenizer.from_pretrained(config.version, subfolder="tokenizer") 
    model_ori = CLIPTextModel.from_pretrained(config.version, subfolder="text_encoder")
    model_ori.eval()
    for param in model_ori.parameters():
        param.requires_grad = False
    model_ori = model_ori.to(config.device)
    
    model_tar = CLIPTextModel.from_pretrained(config.version, subfolder="text_encoder")
    model_tar = model_tar.to(config.device)

    prompts = pd.read_csv(args.input_prompts)

    optimizer = optim.Adam(model_tar.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size,
                                          gamma=config.scheduler_gamma)
    
    criterion = nn.MSELoss()
    
    train(tokenizer, model_ori, model_tar, prompts, optimizer, scheduler, criterion, config, args.save_path)

def train(tokenizer, model_ori, model_tar, df, optimizer, scheduler, criterion, config, save_path):
    model_ori.to(config.device)
    model_tar.to(config.device)
    
    c = [0] * len(df)
    
    need_experience = 1
    experience = 0
    experience_len = 0
    for epoch in range(config.num_epochs):
        epoch_progress_bar = tqdm(total=len(df), desc=f"Epoch [{epoch + 1}/{config.num_epochs}]")
        i = 0
        for _, row in df.iterrows():
            text_r = str(row.prompt_r)
            text_n = str(row.prompt_n)
            text_f = str(row.prompt_f)
            print(f"r:{text_r},n:{text_n},f:{text_f}")

            text_input_r = tokenizer(text_r, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(config.device)
            text_input_n = tokenizer(text_n, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(config.device)
            text_input_f = tokenizer(text_f, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(config.device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                text_features_ori_r = model_ori(text_input_r.input_ids)[0]
                text_features_n = model_ori(text_input_n.input_ids)[0]

            if need_experience == 1:
                with torch.no_grad():
                    text_features_ori_f = model_ori(text_input_f.input_ids)[0]
                c[i] = text_features_ori_f - text_features_ori_r
                experience = experience + c[i]
                experience_len += 1

            text_features_tar_f = model_tar(text_input_f.input_ids)[0]
            text_features_tar_r = model_tar(text_input_r.input_ids)[0]

            loss_forget = criterion(text_features_tar_f, text_features_n)
            loss_retain = criterion(text_features_tar_r, text_features_ori_r)

            similarity = F.cosine_similarity(text_features_tar_f, text_features_ori_r, dim=-1).mean()
            total_loss = loss_retain + config.eta * loss_forget

            total_loss.backward()
            optimizer.step()
            i = i + 1

            epoch_progress_bar.update(1)

        epoch_progress_bar.close()

        if need_experience == 1:
            experience = experience / experience_len
            torch.save(experience, 'experience.pth')
            print("experience saved successfully")
            need_experience = 0

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}], Total_Loss: {total_loss.item()}, Similarity: {similarity.item()}")
        print(f"Loss_Forget: {loss_forget.item()}, Loss_Retain: {loss_retain.item()}")
        print(f"lr: {scheduler.get_last_lr()}")

        if (epoch + 1) % config.save_interval == 0:
            torch.save(model_tar.state_dict(), save_path)
            print(f"Model saved successfully to {save_path}")
        
        #scheduler.step()

if __name__ == "__main__":
    main()
