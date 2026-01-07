import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, interleave_datasets
import threading
from tqdm import tqdm
from queue import Queue
from collections import Counter
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("--num-langs", type=int, default=None, help="Sample N language pairs (default: all)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

model_id = "MaLA-LM/emma-500-llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, legacy=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
base.eval()
for p in base.parameters():
    p.requires_grad = False

torch.cuda.empty_cache()

val_base = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:1")
val_base.eval()
for p in val_base.parameters():
    p.requires_grad = False

class Embedder(nn.Module):
    def __init__(self, base_model, out_dim=768, layer=-4):
        super().__init__()
        self.base = base_model
        self.layer = layer
        hidden = base_model.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[self.layer]                 # [B,T,H]
        mask = attention_mask.unsqueeze(-1)               # [B,T,1]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean pool
        z = self.proj(pooled)
        z = F.normalize(z, p=2, dim=1)

        return z

embedder = Embedder(base, out_dim=768, layer=-4).to(dtype=torch.bfloat16, device="cuda:0").train()
val_embedder = Embedder(val_base, out_dim=768, layer=-4).to(dtype=torch.bfloat16, device="cuda:1").eval()
opt = torch.optim.AdamW(embedder.parameters(), lr=2e-4)

def tokenize(texts, max_length=128, device="cuda:0"):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}

def contrastive_loss(a, b, temp=0.05): # a,b are normalized [N,D]
    logits = (a @ b.T) / temp          # [N,N]
    labels = torch.arange(a.size(0), device=a.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_j) / 2

# All English language pairs - both directions (430 total)
LANG_PAIRS = [
    # X-eng_Latn pairs (122)
    "abk_Cyrl-eng_Latn", "ace_Latn-eng_Latn", "acm_Arab-eng_Latn", "aeb_Arab-eng_Latn",
    "afr_Latn-eng_Latn", "als_Latn-eng_Latn", "amh_Ethi-eng_Latn", "apc_Arab-eng_Latn",
    "ara_Arab-eng_Latn", "arb_Arab-eng_Latn", "arg_Latn-eng_Latn", "arq_Arab-eng_Latn",
    "ars_Arab-eng_Latn", "ary_Arab-eng_Latn", "arz_Arab-eng_Latn", "asm_Beng-eng_Latn",
    "ast_Latn-eng_Latn", "awa_Deva-eng_Latn", "aym_Latn-eng_Latn", "ayr_Latn-eng_Latn",
    "azb_Arab-eng_Latn", "aze_Arab-eng_Latn", "aze_Latn-eng_Latn", "azj_Latn-eng_Latn",
    "bak_Cyrl-eng_Latn", "bam_Latn-eng_Latn", "ban_Latn-eng_Latn", "bar_Latn-eng_Latn",
    "bcl_Latn-eng_Latn", "bel_Cyrl-eng_Latn", "bem_Latn-eng_Latn", "ben_Beng-eng_Latn",
    "bew_Latn-eng_Latn", "bho_Deva-eng_Latn", "bjn_Latn-eng_Latn", "bod_Tibt-eng_Latn",
    "bos_Latn-eng_Latn", "bre_Latn-eng_Latn", "bug_Latn-eng_Latn", "bul_Cyrl-eng_Latn",
    "cat_Latn-eng_Latn", "ceb_Latn-eng_Latn", "ces_Latn-eng_Latn", "chv_Cyrl-eng_Latn",
    "ckb_Arab-eng_Latn", "cmn_Hani-eng_Latn", "crh_Cyrl-eng_Latn", "crh_Latn-eng_Latn",
    "cym_Latn-eng_Latn", "dan_Latn-eng_Latn", "deu_Latn-eng_Latn", "dik_Latn-eng_Latn",
    "diq_Latn-eng_Latn", "dyu_Latn-eng_Latn", "dzo_Tibt-eng_Latn", "ekk_Latn-eng_Latn",
    "ell_Grek-eng_Latn", "epo_Latn-eng_Latn", "est_Latn-eng_Latn", "eus_Latn-eng_Latn",
    "fas_Arab-eng_Latn", "fil_Latn-eng_Latn", "fin_Latn-eng_Latn", "fra_Latn-eng_Latn",
    "fur_Latn-eng_Latn", "glg_Latn-eng_Latn", "grc_Grek-eng_Latn", "gsw_Latn-eng_Latn",
    "hac_Arab-eng_Latn", "hat_Latn-eng_Latn", "heb_Hebr-eng_Latn", "hil_Latn-eng_Latn",
    "hin_Deva-eng_Latn", "hrv_Latn-eng_Latn", "hun_Latn-eng_Latn", "ind_Latn-eng_Latn",
    "isl_Latn-eng_Latn", "ita_Latn-eng_Latn", "jpn_Jpan-eng_Latn", "kab_Latn-eng_Latn",
    "kin_Latn-eng_Latn", "kiu_Latn-eng_Latn", "kor_Hang-eng_Latn", "kor_Zyyy-eng_Latn",
    "lat_Latn-eng_Latn", "lim_Latn-eng_Latn", "lit_Latn-eng_Latn", "ltz_Latn-eng_Latn",
    "lvs_Latn-eng_Latn", "mhr_Cyrl-eng_Latn", "mkd_Cyrl-eng_Latn", "msa_Latn-eng_Latn",
    "nds_Latn-eng_Latn", "nld_Latn-eng_Latn", "nno_Latn-eng_Latn", "nob_Latn-eng_Latn",
    "nor_Latn-eng_Latn", "oci_Latn-eng_Latn", "orv_Cyrl-eng_Latn", "pap_Latn-eng_Latn",
    "pol_Latn-eng_Latn", "por_Latn-eng_Latn", "ron_Latn-eng_Latn", "rus_Cyrl-eng_Latn",
    "sdh_Arab-eng_Latn", "slk_Latn-eng_Latn", "slv_Latn-eng_Latn", "spa_Latn-eng_Latn",
    "srd_Latn-eng_Latn", "srp_Cyrl-eng_Latn", "srp_Latn-eng_Latn", "swe_Latn-eng_Latn",
    "tat_Latn-eng_Latn", "tha_Thai-eng_Latn", "tur_Latn-eng_Latn", "uig_Cyrl-eng_Latn",
    "ukr_Cyrl-eng_Latn", "uzn_Latn-eng_Latn", "vec_Latn-eng_Latn", "vie_Latn-eng_Latn",
    "yao_Latn-eng_Latn", "zsm_Latn-eng_Latn",
    # eng_Latn-X pairs (308)
    "eng_Latn-aba_Latn", "eng_Latn-abk_Cyrl", "eng_Latn-ace_Latn", "eng_Latn-ach_Latn",
    "eng_Latn-aeb_Arab", "eng_Latn-afr_Latn", "eng_Latn-aln_Latn", "eng_Latn-als_Latn",
    "eng_Latn-arb_Arab", "eng_Latn-arg_Latn", "eng_Latn-ars_Arab", "eng_Latn-arz_Arab",
    "eng_Latn-ast_Latn", "eng_Latn-awa_Deva", "eng_Latn-ayr_Latn", "eng_Latn-azb_Arab",
    "eng_Latn-azj_Latn", "eng_Latn-bak_Cyrl", "eng_Latn-bam_Latn", "eng_Latn-ban_Latn",
    "eng_Latn-bar_Latn", "eng_Latn-bcc_Arab", "eng_Latn-bcl_Latn", "eng_Latn-bel_Cyrl",
    "eng_Latn-bem_Latn", "eng_Latn-ben_Beng", "eng_Latn-bew_Latn", "eng_Latn-bho_Deva",
    "eng_Latn-bjn_Latn", "eng_Latn-bre_Latn", "eng_Latn-btx_Latn", "eng_Latn-bul_Cyrl",
    "eng_Latn-cat_Latn", "eng_Latn-cbk_Latn", "eng_Latn-ceb_Latn", "eng_Latn-ces_Latn",
    "eng_Latn-cfm_Latn", "eng_Latn-ckb_Arab", "eng_Latn-cmn_Hani", "eng_Latn-cnh_Latn",
    "eng_Latn-cnr_Latn", "eng_Latn-cos_Latn", "eng_Latn-crh_Latn", "eng_Latn-crs_Latn",
    "eng_Latn-cto_Latn", "eng_Latn-cym_Latn", "eng_Latn-dan_Latn", "eng_Latn-deu_Latn",
    "eng_Latn-ekk_Latn", "eng_Latn-ell_Grek", "eng_Latn-eml_Latn", "eng_Latn-enm_Latn",
    "eng_Latn-epo_Latn", "eng_Latn-est_Latn", "eng_Latn-eus_Latn", "eng_Latn-ewe_Latn",
    "eng_Latn-fao_Latn", "eng_Latn-fas_Arab", "eng_Latn-fij_Latn", "eng_Latn-fil_Latn",
    "eng_Latn-fin_Latn", "eng_Latn-fon_Latn", "eng_Latn-fra_Latn", "eng_Latn-fro_Latn",
    "eng_Latn-frp_Latn", "eng_Latn-fry_Latn", "eng_Latn-fub_Latn", "eng_Latn-ful_Latn",
    "eng_Latn-fur_Latn", "eng_Latn-fuv_Latn", "eng_Latn-gaa_Latn", "eng_Latn-gaz_Latn",
    "eng_Latn-gcf_Latn", "eng_Latn-gla_Latn", "eng_Latn-gle_Latn", "eng_Latn-glg_Latn",
    "eng_Latn-glk_Arab", "eng_Latn-gos_Latn", "eng_Latn-grn_Latn", "eng_Latn-gsw_Latn",
    "eng_Latn-gug_Latn", "eng_Latn-guj_Gujr", "eng_Latn-hat_Latn", "eng_Latn-hau_Latn",
    "eng_Latn-haw_Latn", "eng_Latn-hbo_Hebr", "eng_Latn-hbs_Latn", "eng_Latn-heb_Hebr",
    "eng_Latn-hil_Latn", "eng_Latn-hin_Deva", "eng_Latn-hin_Latn", "eng_Latn-hmr_Latn",
    "eng_Latn-hne_Deva", "eng_Latn-hrv_Latn", "eng_Latn-hrx_Latn", "eng_Latn-hun_Latn",
    "eng_Latn-hye_Armn", "eng_Latn-hyw_Armn", "eng_Latn-ibo_Latn", "eng_Latn-ido_Latn",
    "eng_Latn-ile_Latn", "eng_Latn-ilo_Latn", "eng_Latn-ina_Latn", "eng_Latn-ind_Latn",
    "eng_Latn-isl_Latn", "eng_Latn-ita_Latn", "eng_Latn-jam_Latn", "eng_Latn-jav_Latn",
    "eng_Latn-jbo_Latn", "eng_Latn-jpn_Jpan", "eng_Latn-kab_Latn", "eng_Latn-kac_Latn",
    "eng_Latn-kam_Latn", "eng_Latn-kan_Knda", "eng_Latn-kas_Arab", "eng_Latn-kas_Deva",
    "eng_Latn-kat_Geor", "eng_Latn-kau_Zyyy", "eng_Latn-kaz_Cyrl", "eng_Latn-kbp_Latn",
    "eng_Latn-kea_Latn", "eng_Latn-khk_Cyrl", "eng_Latn-khm_Khmr", "eng_Latn-kik_Latn",
    "eng_Latn-kin_Latn", "eng_Latn-kir_Cyrl", "eng_Latn-kiu_Latn", "eng_Latn-kjh_Cyrl",
    "eng_Latn-kmb_Latn", "eng_Latn-kmr_Latn", "eng_Latn-kng_Latn", "eng_Latn-kon_Latn",
    "eng_Latn-kor_Hang", "eng_Latn-kor_Zyyy", "eng_Latn-ksh_Latn", "eng_Latn-ktu_Latn",
    "eng_Latn-kwy_Latn", "eng_Latn-lad_Latn", "eng_Latn-lao_Laoo", "eng_Latn-lat_Latn",
    "eng_Latn-lav_Latn", "eng_Latn-lfn_Latn", "eng_Latn-lij_Latn", "eng_Latn-lim_Latn",
    "eng_Latn-lin_Latn", "eng_Latn-lit_Latn", "eng_Latn-lmo_Latn", "eng_Latn-ltg_Latn",
    "eng_Latn-ltz_Latn", "eng_Latn-lua_Latn", "eng_Latn-lub_Latn", "eng_Latn-lug_Latn",
    "eng_Latn-luo_Latn", "eng_Latn-lus_Latn", "eng_Latn-lus_Zyyy", "eng_Latn-lvs_Latn",
    "eng_Latn-lzh_Hani", "eng_Latn-mad_Latn", "eng_Latn-mag_Deva", "eng_Latn-mai_Deva",
    "eng_Latn-mal_Mlym", "eng_Latn-mar_Deva", "eng_Latn-mar_Latn", "eng_Latn-mav_Latn",
    "eng_Latn-mfe_Latn", "eng_Latn-mhr_Cyrl", "eng_Latn-min_Latn", "eng_Latn-mkd_Cyrl",
    "eng_Latn-mlg_Latn", "eng_Latn-mlt_Latn", "eng_Latn-mni_Beng", "eng_Latn-mon_Cyrl",
    "eng_Latn-mos_Latn", "eng_Latn-mri_Latn", "eng_Latn-mrj_Cyrl", "eng_Latn-msa_Latn",
    "eng_Latn-mui_Latn", "eng_Latn-mwl_Latn", "eng_Latn-mya_Mymr", "eng_Latn-nah_Latn",
    "eng_Latn-nap_Latn", "eng_Latn-nbl_Latn", "eng_Latn-nde_Latn", "eng_Latn-nds_Latn",
    "eng_Latn-nep_Deva", "eng_Latn-nld_Latn", "eng_Latn-nno_Latn", "eng_Latn-nob_Latn",
    "eng_Latn-nor_Latn", "eng_Latn-npi_Deva", "eng_Latn-nso_Latn", "eng_Latn-nus_Latn",
    "eng_Latn-nya_Latn", "eng_Latn-nzi_Latn", "eng_Latn-oci_Latn", "eng_Latn-ori_Orya",
    "eng_Latn-orm_Latn", "eng_Latn-orv_Cyrl", "eng_Latn-ory_Orya", "eng_Latn-pag_Latn",
    "eng_Latn-pam_Latn", "eng_Latn-pan_Guru", "eng_Latn-pan_Latn", "eng_Latn-pap_Latn",
    "eng_Latn-pbt_Arab", "eng_Latn-pcd_Latn", "eng_Latn-pcm_Latn", "eng_Latn-plt_Latn",
    "eng_Latn-pms_Latn", "eng_Latn-pnb_Arab", "eng_Latn-pol_Latn", "eng_Latn-por_Latn",
    "eng_Latn-prs_Arab", "eng_Latn-pus_Arab", "eng_Latn-qub_Latn", "eng_Latn-que_Latn",
    "eng_Latn-quh_Latn", "eng_Latn-quw_Latn", "eng_Latn-quy_Latn", "eng_Latn-quz_Latn",
    "eng_Latn-rmn_Latn", "eng_Latn-rmy_Latn", "eng_Latn-rnd_Latn", "eng_Latn-roh_Latn",
    "eng_Latn-ron_Latn", "eng_Latn-run_Latn", "eng_Latn-rus_Cyrl", "eng_Latn-sag_Latn",
    "eng_Latn-sah_Cyrl", "eng_Latn-san_Deva", "eng_Latn-scn_Latn", "eng_Latn-sco_Latn",
    "eng_Latn-shn_Mymr", "eng_Latn-shp_Latn", "eng_Latn-sin_Sinh", "eng_Latn-slk_Latn",
    "eng_Latn-slv_Latn", "eng_Latn-sme_Latn", "eng_Latn-smo_Latn", "eng_Latn-sna_Latn",
    "eng_Latn-snd_Arab", "eng_Latn-som_Latn", "eng_Latn-sot_Latn", "eng_Latn-spa_Latn",
    "eng_Latn-sqi_Latn", "eng_Latn-srd_Latn", "eng_Latn-srp_Cyrl", "eng_Latn-srp_Latn",
    "eng_Latn-ssw_Latn", "eng_Latn-sun_Latn", "eng_Latn-swa_Latn", "eng_Latn-swc_Latn",
    "eng_Latn-swe_Latn", "eng_Latn-swh_Latn", "eng_Latn-szl_Latn", "eng_Latn-tam_Taml",
    "eng_Latn-taq_Latn", "eng_Latn-tar_Latn", "eng_Latn-tat_Cyrl", "eng_Latn-tel_Latn",
    "eng_Latn-tel_Telu", "eng_Latn-tgk_Cyrl", "eng_Latn-tgl_Latn", "eng_Latn-tha_Thai",
    "eng_Latn-tig_Ethi", "eng_Latn-tir_Ethi", "eng_Latn-tlh_Latn", "eng_Latn-tpi_Latn",
    "eng_Latn-tsn_Latn", "eng_Latn-tso_Latn", "eng_Latn-tuk_Arab", "eng_Latn-tuk_Latn",
    "eng_Latn-tum_Latn", "eng_Latn-tur_Latn", "eng_Latn-twi_Latn", "eng_Latn-tzm_Tfng",
    "eng_Latn-uig_Arab", "eng_Latn-uig_Latn", "eng_Latn-ukr_Cyrl", "eng_Latn-umb_Latn",
    "eng_Latn-urd_Arab", "eng_Latn-urd_Latn", "eng_Latn-uzb_Latn", "eng_Latn-uzn_Cyrl",
    "eng_Latn-uzn_Latn", "eng_Latn-vec_Latn", "eng_Latn-ven_Latn", "eng_Latn-vie_Latn",
    "eng_Latn-vls_Latn", "eng_Latn-war_Latn", "eng_Latn-wes_Latn", "eng_Latn-wln_Latn",
    "eng_Latn-wol_Latn", "eng_Latn-wuu_Hani", "eng_Latn-xav_Latn", "eng_Latn-xho_Latn",
    "eng_Latn-ydd_Hebr", "eng_Latn-yid_Hebr", "eng_Latn-yor_Latn", "eng_Latn-yue_Hani",
    "eng_Latn-zea_Latn", "eng_Latn-zho_Hani", "eng_Latn-zho_Hans", "eng_Latn-zho_Hant",
    "eng_Latn-zho_Zyyy", "eng_Latn-zpa_Latn", "eng_Latn-zsm_Latn", "eng_Latn-zul_Latn",
]

# Sample language pairs if requested
random.seed(args.seed)
if args.num_langs is not None:
    selected_pairs = random.sample(LANG_PAIRS, min(args.num_langs, len(LANG_PAIRS)))
    print(f"Sampled {len(selected_pairs)} language pairs:")
    for lp in selected_pairs:
        print(f"  {lp}")
else:
    selected_pairs = LANG_PAIRS
    print(f"Using all {len(selected_pairs)} language pairs")

# Load and interleave selected pairs
datasets = [
    load_dataset("MaLA-LM/FineOPUS-ReLID", data_dir=lp, split="train", streaming=True)
    for lp in tqdm(selected_pairs, desc="Loading language pairs")
]
dataset = interleave_datasets(datasets)
dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)

max_batch_tokens = 1024  # total tokens per batch (across all samples)
max_length = 128
val_size = 32

# Create fixed validation set from a single pair (faster to load)
print("Creating validation set from eng_Latn-deu_Latn...")
val_dataset = load_dataset("MaLA-LM/FineOPUS-ReLID", data_dir="eng_Latn-deu_Latn", split="train", streaming=True)
val_iter = iter(val_dataset)
val_left, val_right = [], []
for _ in tqdm(range(val_size), desc="Validation samples"):
    sample = next(val_iter)
    val_left.append(sample["source_text"])
    val_right.append(sample["target_text"])
val_batch_l = tokenize(val_left, device="cuda:1")
val_batch_r = tokenize(val_right, device="cuda:1")

# Now create training iterator
data_iter = iter(dataset)

# Async validation
val_thread = None
val_loss_result = [None]  # use list for mutability in thread

def run_validation():
    # Copy projection weights to cuda:1
    state = {k: v.to("cuda:1") for k, v in embedder.proj.state_dict().items()}
    val_embedder.proj.load_state_dict(state)
    with torch.no_grad():
        val_a = val_embedder(**val_batch_l)
        val_b = val_embedder(**val_batch_r)
        loss = contrastive_loss(val_a, val_b)
    val_loss_result[0] = loss.item()

def estimate_tokens(text, max_len):
    """Fast token estimate: ~4 chars per token for LLaMA."""
    return min(len(text) // 4 + 1, max_len)

# Language pair statistics
lang_pair_counts = Counter()
lang_pair_lock = threading.Lock()

def get_batch(data_iter, max_tokens, max_len):
    """Accumulate samples until we hit the token budget."""
    left, right = [], []
    total_tokens = 0
    batch_counts = Counter()
    while total_tokens < max_tokens:
        sample = next(data_iter)
        src, tgt = sample["source_text"], sample["target_text"]
        # Track language pair (use original labels, not detected)
        lang_pair = f"{sample['orig_src_lang']}-{sample['orig_tgt_lang']}"
        batch_counts[lang_pair] += 1
        total_tokens += max(estimate_tokens(src, max_len), estimate_tokens(tgt, max_len))
        left.append(src)
        right.append(tgt)
    # Update global counter
    with lang_pair_lock:
        lang_pair_counts.update(batch_counts)
    return left, right

def print_lang_stats(top_n=10):
    """Print top N language pairs seen during training."""
    with lang_pair_lock:
        total = sum(lang_pair_counts.values())
        print(f"\nLanguage pair stats ({total} samples, {len(lang_pair_counts)} pairs):")
        for pair, count in lang_pair_counts.most_common(top_n):
            print(f"  {pair}: {count} ({100*count/total:.1f}%)")
        print()

# Prefetch batches in background
batch_queue = Queue(maxsize=4)
stop_prefetch = threading.Event()

def prefetch_worker():
    while not stop_prefetch.is_set():
        try:
            left, right = get_batch(data_iter, max_batch_tokens, max_length)
            batch_l = tokenize(left, max_length=max_length, device="cuda:0")
            batch_r = tokenize(right, max_length=max_length, device="cuda:0")
            batch_queue.put((batch_l, batch_r))
        except StopIteration:
            break

prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
prefetch_thread.start()

pbar = tqdm(range(1000), desc="Training")
for step in pbar:
    batch_l, batch_r = batch_queue.get()

    a = embedder(**batch_l)
    b = embedder(**batch_r)

    loss = contrastive_loss(a, b)

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Update progress bar with train loss
    pbar.set_postfix(train=f"{loss.detach().item():.4f}", val=f"{val_loss_result[0]:.4f}" if val_loss_result[0] else "...")

    if step % 100 == 0:
        # Wait for previous validation if running
        if val_thread is not None:
            val_thread.join()
        # Launch new validation on cuda:1
        val_thread = threading.Thread(target=run_validation)
        val_thread.start()
        # For step 0, wait for result
        if step == 0:
            val_thread.join()
            pbar.set_postfix(train=f"{loss.detach().item():.4f}", val=f"{val_loss_result[0]:.4f}")
        # Save checkpoint
        torch.save(embedder.state_dict(), f"embedder_step{step}.pt")
        # Print language stats
        print_lang_stats()

# Stop prefetch worker
stop_prefetch.set()

# Wait for final validation
if val_thread is not None:
    val_thread.join()

# Save the model
torch.save(embedder.state_dict(), "embedder.pt")
print("Model saved to embedder.pt")
