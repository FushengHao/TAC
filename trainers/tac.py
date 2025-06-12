import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from kmeans_pytorch import kmeans
import json

_tokenizer = _Tokenizer()


dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.TAC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.TAC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.TAC.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.TAC.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_ctx = n_ctx

    def forward(self, prompts, tokenized_prompts, topk=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, topk=topk)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1) + self.n_ctx] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.TAC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.TAC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.TAC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 20:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.TAC.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
                
        with open(f"./descriptions/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
            
        with torch.no_grad():
            clip_weights = []
            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace("_", " ")
                texts = []
                for t in gpt3_prompt[classname]:
                    texts.append(t)
                texts = clip.tokenize(texts)
                if torch.cuda.is_available():
                    texts = texts.cuda()
                class_embeddings = clip_model_temp.encode_text(texts)
                class_embeddings = class_embeddings.mean(dim=0, keepdim=True)
                clip_weights.append(class_embeddings)
                
        clip_weights = torch.cat(clip_weights, dim=0)
        clip_weights = clip_weights / clip_weights.norm(dim=-1, keepdim=True)
        self.fixed_embeddings = clip_weights
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1:-n_ctx-5, :])  # CLS, EOS
        
        with torch.no_grad():
            _, cluster_centers = kmeans(X=clip_weights, num_clusters=5, distance='cosine', device=clip_weights.device)
            cluster_centers = cluster_centers.half().to(clip_weights.device)
            cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
            self.topk = nn.Embedding.from_pretrained(cluster_centers).weight
            
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, topk_tokens, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                topk_tokens,
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, topk):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        topk_tokens = topk.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = self.construct_prompts(ctx, prefix, suffix, topk_tokens)

        return prompts


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.topk = self.prompt_learner.topk
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, cfg.TRAINER.TAC.N_CTX_TEXT + 5)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        self.k_transforms = cfg.DATALOADER.K_TRANSFORMS
        
        self.VPT_topk_t = nn.Linear(clip_model.visual.output_dim, clip_model.visual.output_dim).half()
        self.VPT_topk_v = nn.Linear(clip_model.visual.output_dim, clip_model.visual.width).half()

    def forward(self, image, label=None, img_o=None):
        tokenized_prompts = self.tokenized_prompts
        topk_t = self.VPT_topk_t(self.topk)
        topk_v = self.VPT_topk_v(self.topk)
        logit_scale = self.logit_scale.exp()
            
        fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features

        prompts = self.prompt_learner(topk_t)
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts, topk=topk_t)
        image_features = self.image_encoder(image.type(self.dtype), topk=topk_v)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(img_o.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                zero_shot_features = zero_shot_features.repeat(self.k_transforms, 1)

            return logits, text_features, fixed_embeddings, zero_shot_features, image_features
        else:
            return logits


@TRAINER_REGISTRY.register()
class TAC(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.TAC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.TAC.PREC == "fp32" or cfg.TRAINER.TAC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        self.ep_model_weights = np.array([1.0 for a in range(1, N + 1)])
        self.ep_model_weights = self.ep_model_weights / sum(self.ep_model_weights)
        self.scaler = GradScaler() if cfg.TRAINER.TAC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
            self.model.image_encoder = nn.DataParallel(self.model.image_encoder)
        
        self.previous_model_weight = None
        
        self.k_transforms = cfg.DATALOADER.K_TRANSFORMS
        self.n_cls = len(classnames)

    def forward_backward(self, batch):
        image, label, img_o = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        label = label.repeat(self.k_transforms)

        prec = self.cfg.TRAINER.TAC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, img_o)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft = model(image, label, img_o)
            
            loss_ce = F.cross_entropy(logits, label)
            
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                      reduction='mean') * self.cfg.TRAINER.TAC.TEXT_LOSS_WEIGHT
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean') * self.cfg.TRAINER.TAC.IMAGE_LOSS_WEIGHT

            L_SCL = loss_scl_text + loss_scl_image
            loss = loss_ce + L_SCL
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss_ce": loss_ce.item(), "loss_scl_text": loss_scl_text.item(), "loss_scl_image": loss_scl_image.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.ep_model_weights[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_weight is None:
                self.previous_model_weight = weighted_state_dict
            else:
                self.previous_model_weight = self.state_dict_add(weighted_state_dict, self.previous_model_weight)

        if self.step_counter == self.model.total_epochs + 1:
            print("Using weighted model for final inference...")
            model.load_state_dict(self.previous_model_weight)
            self.model.load_state_dict(self.previous_model_weight)
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        img_o = batch["img0"]
        
        if isinstance(input, (tuple, list)):
            input = torch.cat(input, dim=0)
        
        input = input.to(self.device)
        label = label.to(self.device)
        img_o = img_o.to(self.device)
        return input, label, img_o

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)