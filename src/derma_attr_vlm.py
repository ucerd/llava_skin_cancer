"""
DermaAttr-VLM: implementation of the architecture described in the manuscript,
built on the real local LLaVA-1.5-7B (4-bit QLoRA) backbone.

Pathways (paper Sec. Model Architecture):
  - Visual encoder: CLIP ViT-L/14-336 patch tokens V in R^{Npatch x 1024} (layer -2, CLS dropped).
  - Structured attribute embedding: grouped linear maps a -> U in R^{m x 1024}.
  - Gated cross-modal fusion: V* = V + gamma * Attn(V, U), gamma init ~0.
  - Visual projection: Vhat = LN(MLP(V*)) + delta * V* Ws  -> 4096 (MLP = pretrained LLaVA projector).
  - Attribute projection: Uhat = U P -> 4096.
  - Multimodal sequence: [IMG Vhat][ATTR Uhat][TEXT] -> LLaMA/Vicuna decoder + LoRA.
  - Multi-task heads: classification (malignant/benign), attribute reconstruction.
  - Loss: L = L_LM + lam_cls L_CLS + lam_rec L_REC + lam_dist L_DIST.

Ablation flags:
  use_attr, attr_source in {oracle, predicted, shuffled, none}, use_gate, use_rec, use_dist.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ATTRS = ["asymmetry", "border_irregularity", "color_variegation", "num_colors",
         "pigment_network", "dots_globules", "streaks", "blue_white_veil",
         "regression_structures", "vascular_patterns", "ulceration_crusting"]
D_ATTR = len(ATTRS)
# clinically motivated groups over the primary attribute set (paper: G groups)
ATTR_GROUPS = [
    [0, 1],           # shape: asymmetry, border
    [2, 3],           # color: variegation, num_colors
    [4, 5, 6],        # structural: pigment network, dots/globules, streaks
    [7, 8, 9, 10],    # dermoscopic patterns: veil, regression, vascular, ulceration
]
TOKENS_PER_GROUP = 2  # m = 4 groups * 2 = 8 attribute tokens


class AttributeEncoder(nn.Module):
    """Grouped linear maps producing m attribute tokens in the 1024-d visual space."""
    def __init__(self, dim=1024, groups=ATTR_GROUPS, tpg=TOKENS_PER_GROUP):
        super().__init__()
        self.groups = groups
        self.tpg = tpg
        self.dim = dim
        self.maps = nn.ModuleList([nn.Linear(len(g), tpg * dim) for g in groups])

    def forward(self, a):  # a: (B, D_ATTR)
        toks = []
        for gi, g in enumerate(self.groups):
            u = self.maps[gi](a[:, g])              # (B, tpg*dim)
            toks.append(u.view(a.size(0), self.tpg, self.dim))
        return torch.cat(toks, dim=1)               # (B, m, dim)


class GatedCrossFusion(nn.Module):
    """V* = V + gamma * MHA(query=V, key=value=U). gamma init ~0."""
    def __init__(self, dim=1024, heads=8, use_gate=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.use_gate = use_gate
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, V, U):
        att, _ = self.attn(self.norm_q(V), self.norm_kv(U), self.norm_kv(U))
        g = torch.tanh(self.gamma) if self.use_gate else 1.0
        return V + g * att


class AttrHead(nn.Module):
    """Predicts attribute vector a_hat from pooled visual features (predicted-attr pathway)."""
    def __init__(self, dim=1024, d_attr=D_ATTR):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, d_attr))

    def forward(self, v_pool):
        return torch.sigmoid(self.net(v_pool))      # attrs are in [0,1]


class DermaAttrVLM(nn.Module):
    def __init__(self, peft_model, tokenizer, cfg):
        """
        peft_model: get_peft_model(LlavaForConditionalGeneration, lora_cfg)
        cfg: dict with keys use_attr, attr_source, use_gate, use_rec, use_dist,
             lam_cls, lam_rec, lam_dist
        """
        super().__init__()
        self.peft = peft_model
        self.llava = peft_model.base_model.model            # underlying LlavaForConditionalGeneration
        self.tok = tokenizer
        self.cfg = cfg
        dim_v = 1024
        dim_l = self.llava.config.text_config.hidden_size   # 4096
        self.dtype = next(p for p in self.llava.parameters()).dtype

        # trainable modules (kept in fp32 for stability; cast at fusion boundary)
        self.attr_enc = AttributeEncoder(dim_v)
        self.fusion = GatedCrossFusion(dim_v, use_gate=cfg.get("use_gate", True))
        self.attr_head = AttrHead(dim_v)
        self.Ws = nn.Parameter(torch.zeros(dim_v, dim_l))   # gated residual (delta*V*Ws)
        self.delta = nn.Parameter(torch.zeros(1))
        self.ln_vis = nn.LayerNorm(dim_l)
        self.attr_proj = nn.Linear(dim_v, dim_l)            # P: attribute -> language space
        self.cls_head = nn.Sequential(nn.LayerNorm(dim_v), nn.Linear(dim_v, 256),
                                      nn.GELU(), nn.Linear(256, 2))
        self.rec_head = nn.Sequential(nn.LayerNorm(dim_v), nn.Linear(dim_v, 256),
                                      nn.GELU(), nn.Linear(256, D_ATTR))
        self.register_buffer("class_weights", torch.ones(2), persistent=False)

    # ---- vision ----
    def raw_patch_features(self, pixel_values):
        vt = self.llava.model.vision_tower
        out = vt(pixel_values.to(self.dtype), output_hidden_states=True)
        feat = out.hidden_states[self.llava.config.vision_feature_layer]  # (B, 1+N, 1024)
        if self.llava.config.vision_feature_select_strategy == "default":
            feat = feat[:, 1:]
        return feat  # (B, N, 1024)

    def project_visual(self, Vstar):
        mlp = self.llava.model.multi_modal_projector(Vstar.to(self.dtype)).float()  # (B,N,4096)
        res = self.delta * (Vstar.float() @ self.Ws)
        return self.ln_vis(mlp) + res

    # ---- attribute assembly given a source ----
    def build_attr(self, a_ref, v_pool, a_control=None):
        src = self.cfg.get("attr_source", "oracle")
        a_hat = self.attr_head(v_pool.float())
        if src == "predicted":
            a_used = a_hat
        elif src == "shuffled":
            if a_control is None:
                raise ValueError("The shuffled condition requires a dataset-level control attribute.")
            a_used = a_control
        elif src == "none":
            a_used = None
        else:  # oracle
            a_used = a_ref
        return a_used, a_hat

    def forward(self, pixel_values, a_ref, y_cls,
                prefix_ids, mid_ids, ans_ids, mid_lens, ans_lens, a_control=None):
        """
        Text is pre-split into three id chunks per sample (padded tensors + lengths):
          prefix_ids: "USER: " (same for all, incl BOS)  -> (Lp,)
          mid_ids:    question + "\nASSISTANT: "           -> (B, Lm) padded
          ans_ids:    answer + EOS                          -> (B, La) padded
        Image and attribute embeds are spliced between prefix and mid.
        """
        B = pixel_values.size(0)
        dev = pixel_values.device
        V = self.raw_patch_features(pixel_values).float()          # (B,N,1024)
        v_pool = V.mean(1)                                          # (B,1024)

        use_attr = self.cfg.get("use_attr", True) and self.cfg.get("attr_source") != "none"
        a_used, a_hat = self.build_attr(a_ref, v_pool, a_control=a_control)

        if use_attr and a_used is not None:
            U = self.attr_enc(a_used.float())                      # (B,m,1024)
            Vstar = self.fusion(V, U)                              # (B,N,1024)
            attr_embeds = self.attr_proj(U.to(self.dtype)).float() # (B,m,4096)
        else:
            U = None
            Vstar = V
            attr_embeds = None

        img_embeds = self.project_visual(Vstar)                    # (B,N,4096)
        vis_only_embeds = self.project_visual(V) if self.cfg.get("use_dist", True) else None

        emb = self.peft.get_input_embeddings()
        pref = emb(prefix_ids.to(dev)).float()                     # (Lp,4096)
        Lp = pref.size(0)
        Nimg = img_embeds.size(1)
        m = attr_embeds.size(1) if attr_embeds is not None else 0

        seqs, labels, masks = [], [], []
        for i in range(B):
            parts = [pref, img_embeds[i]]
            if attr_embeds is not None:
                parts.append(attr_embeds[i])
            mid = emb(mid_ids[i, :mid_lens[i]].to(dev)).float()
            ans = emb(ans_ids[i, :ans_lens[i]].to(dev)).float()
            parts += [mid, ans]
            s = torch.cat(parts, 0)
            ctx = Lp + Nimg + m + mid.size(0)
            lab = torch.full((s.size(0),), -100, dtype=torch.long, device=dev)
            lab[ctx:] = ans_ids[i, :ans_lens[i]].to(dev)
            seqs.append(s); labels.append(lab)
            masks.append(torch.ones(s.size(0), dtype=torch.long, device=dev))

        L = max(s.size(0) for s in seqs)
        embeds = torch.zeros(B, L, img_embeds.size(-1), device=dev)
        lab_t = torch.full((B, L), -100, dtype=torch.long, device=dev)
        am = torch.zeros(B, L, dtype=torch.long, device=dev)
        for i, (s, lab, mk) in enumerate(zip(seqs, labels, masks)):
            n = s.size(0)
            embeds[i, :n] = s; lab_t[i, :n] = lab; am[i, :n] = mk

        out = self.peft(inputs_embeds=embeds.to(self.dtype), attention_mask=am, labels=lab_t)
        loss = out.loss

        # multi-task heads
        logs = {}
        cls_logits = self.cls_head(Vstar.mean(1).float())
        valid = y_cls >= 0
        if valid.any():
            cls_loss = F.cross_entropy(
                cls_logits[valid], y_cls[valid].to(dev), weight=self.class_weights
            )
            loss = loss + self.cfg.get("lam_cls", 0.1) * cls_loss
            logs["cls"] = float(cls_loss.detach())
        if self.cfg.get("use_rec", True):
            rec = self.rec_head(v_pool)
            rec_loss = F.mse_loss(rec, a_ref.float())
            loss = loss + self.cfg.get("lam_rec", 0.05) * rec_loss
            logs["rec"] = float(rec_loss.detach())
        # attribute-head supervision (so predicted pathway is learnable)
        ah_loss = F.mse_loss(a_hat, a_ref.float())
        loss = loss + 0.05 * ah_loss
        logs["ahead"] = float(ah_loss.detach())
        if vis_only_embeds is not None and use_attr and attr_embeds is not None:
            dist_loss = F.mse_loss(img_embeds, vis_only_embeds.detach())
            loss = loss + self.cfg.get("lam_dist", 0.02) * dist_loss
            logs["dist"] = float(dist_loss.detach())

        logs["lm"] = float(out.loss.detach())
        return loss, cls_logits.detach(), logs

    @torch.no_grad()
    def classify(self, pixel_values, a_ref, a_control=None):
        V = self.raw_patch_features(pixel_values).float()
        v_pool = V.mean(1)
        a_used, _ = self.build_attr(a_ref, v_pool, a_control=a_control)
        use_attr = self.cfg.get("use_attr", True) and self.cfg.get("attr_source") != "none"
        if use_attr and a_used is not None:
            U = self.attr_enc(a_used.float())
            Vstar = self.fusion(V, U)
        else:
            Vstar = V
        return F.softmax(self.cls_head(Vstar.mean(1).float()), dim=-1)[:, 1]

    @torch.no_grad()
    def generate(self, pixel_values, a_ref, prefix_ids, mid_ids, mid_len,
                 max_new_tokens=96, a_control=None):
        dev = pixel_values.device
        V = self.raw_patch_features(pixel_values).float()
        v_pool = V.mean(1)
        use_attr = self.cfg.get("use_attr", True) and self.cfg.get("attr_source") != "none"
        a_used, _ = self.build_attr(a_ref, v_pool, a_control=a_control)
        if use_attr and a_used is not None:
            U = self.attr_enc(a_used.float())
            Vstar = self.fusion(V, U)
            attr_embeds = self.attr_proj(U.to(self.dtype)).float()
        else:
            Vstar = V; attr_embeds = None
        img_embeds = self.project_visual(Vstar)
        emb = self.peft.get_input_embeddings()
        pref = emb(prefix_ids.to(dev)).float()
        parts = [pref, img_embeds[0]]
        if attr_embeds is not None:
            parts.append(attr_embeds[0])
        parts.append(emb(mid_ids[0, :mid_len].to(dev)).float())
        seq = torch.cat(parts, 0).unsqueeze(0).to(self.dtype)
        am = torch.ones(1, seq.size(1), dtype=torch.long, device=dev)
        gen = self.peft.generate(inputs_embeds=seq, attention_mask=am,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(gen[0], skip_special_tokens=True)

    def trainable_modules(self):
        return [self.attr_enc, self.fusion, self.attr_head, self.ln_vis,
                self.attr_proj, self.cls_head, self.rec_head]
