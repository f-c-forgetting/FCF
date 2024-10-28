import argparse
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

# Configuration class
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
        self.eta_clean = 0.7

class TextFeatureFilter:
    def __init__(self, version, tokenizer, device='cuda'):
        self.device = device
        self.model = CLIPTextModel.from_pretrained(version, subfolder="text_encoder").to(self.device)
        self.model.eval()

    def encode_texts(self, texts):
        with torch.no_grad():
            text_features = self.model.encode(texts).to(self.device)
        return text_features

    def compute_mean_feature(self, texts, tokenizer):
        text_input = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model(text_input.input_ids.to(self.device))[0]
        return features.mean(dim=0)

    def remove_projection(self, tokenizer, target_texts, feature_texts, eta_clean):
        feature_mean = self.compute_mean_feature(feature_texts, tokenizer)
        target_features = self.compute_mean_feature(target_texts, tokenizer)

        feature_mean_norm = feature_mean / feature_mean.norm()

        projection_length = torch.sum(target_features * feature_mean_norm)
        projection = projection_length * feature_mean_norm

        weighted_projection = eta_clean * projection
        cleaned_features = target_features - weighted_projection.to(self.device)
        cleaned_features = cleaned_features / (1 - eta_clean)

        return cleaned_features

def main():
    parser = argparse.ArgumentParser(description='Text Feature Filter Training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()

    config = Config()
    tokenizer = CLIPTokenizer.from_pretrained(config.version, subfolder="tokenizer")
    model_ori = CLIPTextModel.from_pretrained(config.version, subfolder="text_encoder").to(config.device)
    model_ori.eval()
    for param in model_ori.parameters():
        param.requires_grad = False
    model_tar = CLIPTextModel.from_pretrained(config.version, subfolder="text_encoder").to(config.device)
    for param in model_tar.parameters():
        param.requires_grad = True

    text_filter = TextFeatureFilter(config.version, tokenizer, device=config.device)

    for i in range(2):
        model_path = args.model_path  # Use the model path from command line argument
        model_tar.load_state_dict(torch.load(model_path))
        if (i == 0):
            needclean_texts = ["male", "boy", "man"]
        elif (i == 1):
            needclean_texts = ["female", "girl", "woman"]

        concept_texts = ["naked", "nudity"]
        retain_text = ["people is wearing a beautiful clothe."]

        optimizer = optim.Adam(model_tar.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        criterion = nn.MSELoss()

        cleaned_features = text_filter.remove_projection(tokenizer, needclean_texts, concept_texts, config.eta_clean)

        for epoch in range(config.num_epochs):
            optimizer.zero_grad()

            with torch.no_grad():
                put_retain_text = tokenizer(retain_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(config.device)
                text_features_ori_r = model_ori(put_retain_text.input_ids)[0]
                text_features_n = cleaned_features

            put_retain_text = tokenizer(retain_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(config.device)
            put_needclean_texts = tokenizer(needclean_texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(config.device)
            text_features_tar_f = model_tar(put_needclean_texts.input_ids)[0]
            text_features_tar_r = model_tar(put_retain_text.input_ids)[0]

            loss_forget = criterion(text_features_tar_f, text_features_n)
            loss_retain = criterion(text_features_tar_r, text_features_ori_r)

            similarity = F.cosine_similarity(text_features_tar_f, text_features_ori_r, dim=-1).mean()
            total_loss = loss_retain + config.eta_forget * loss_forget
            total_loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{config.num_epochs}], Total_Loss: {total_loss.item()}, Similarity: {similarity.item()}")
            print(f"Loss_Forget: {loss_forget.item()}, Loss_Retain: {loss_retain.item()}")
            print(f"lr: {scheduler.get_last_lr()}")

            if (epoch + 1) % config.save_interval == 0:
                torch.save(model_tar.state_dict(), model_path)
                print("model saved successfully")

    model_tar.eval()

if __name__ == "__main__":
    main()
