import argparse
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel


class Config:
    def __init__(self):
        self.version = 'CompVis/stable-diffusion-v1-4'
        self.device = torch.device("cuda")
        self.max_length = 77
        self.n_repeat = 1
        self.normalize = True
        self.learning_rate = 0.000025
        self.num_epochs = 60
        self.save_interval = 60
        self.scheduler_step_size = 3
        self.scheduler_gamma = 0.1
        self.eta_forget = 0.25
        self.eta_experience = 1



def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train CLIP model with custom settings.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the checkpoint model file.')
    parser.add_argument('--experience_path', type=str, required=True,
                        help='Path to the nudity experience file.')

    args = parser.parse_args()

    config = Config()
    tokenizer = CLIPTokenizer.from_pretrained(config.version, subfolder="tokenizer")
    model_ori = CLIPTextModel.from_pretrained(config.version, subfolder="text_encoder").to(config.device)
    model_ori.eval()
    for param in model_ori.parameters():
        param.requires_grad = False
    model_tar = CLIPTextModel.from_pretrained(config.version, subfolder="text_encoder").to(config.device)
    for param in model_ori.parameters():
        param.requires_grad = True

    #################################
    for i in range(2):
        model_path = args.model_path
        model_tar.load_state_dict(torch.load(model_path))
        nudity_experience = torch.load(args.experience_path)

        if i == 0:
            needclean_text = ["male", "boy", "man"]
        elif i == 1:
            needclean_text = ["female", "girl", "woman"]

        retain_text = ["people is wearing a beautiful clothe."]
        optimizer = optim.Adam(model_tar.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size,
                                              gamma=config.scheduler_gamma)
        criterion = nn.MSELoss()

        for epoch in range(config.num_epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                put_retain_text = tokenizer(retain_text, padding="max_length", max_length=tokenizer.model_max_length,
                                            truncation=True, return_tensors="pt").to(config.device)
                put_needclean_text = tokenizer(needclean_text, padding="max_length",
                                               max_length=tokenizer.model_max_length, truncation=True,
                                               return_tensors="pt").to(config.device)
                text_features_ori_r = model_ori(put_retain_text.input_ids)[0]
                text_features_n = model_ori(put_needclean_text.input_ids)[0] - config.eta_experience * nudity_experience

            put_retain_text = tokenizer(retain_text, padding="max_length", max_length=tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt").to(config.device)
            put_needclean_text = tokenizer(needclean_text, padding="max_length", max_length=tokenizer.model_max_length,
                                           truncation=True, return_tensors="pt").to(config.device)
            text_features_tar_f = model_tar(put_needclean_text.input_ids)[0]
            text_features_tar_r = model_tar(put_retain_text.input_ids)[0]

            loss_forget = criterion(text_features_tar_f, text_features_n)
            loss_retain = criterion(text_features_tar_r, text_features_ori_r)

            similarity = F.cosine_similarity(text_features_tar_f, text_features_ori_r, dim=-1).mean()
            total_loss = loss_retain + config.eta_forget * loss_forget

            total_loss.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}], Total_Loss: {total_loss.item()}, Similarity: {similarity.item()}")
            print(f"Loss_Forget: {loss_forget.item()}, Loss_Retain: {loss_retain.item()}")
            print(f"lr: {scheduler.get_last_lr()}")

            if (epoch + 1) % config.save_interval == 0:
                torch.save(model_tar.state_dict(), model_path)
                print("model saved successfully")

    model_tar.eval()


if __name__ == "__main__":
    main()
