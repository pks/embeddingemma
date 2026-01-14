#!/usr/bin/env python3
"""Fine-tune multilingual embeddings using contrastive learning."""

import torch
import torch.nn.functional as F
from datasets import load_dataset, interleave_datasets
import threading
from tqdm import tqdm
from queue import Queue
from collections import Counter
import argparse
import random
import os
import langcodes

from common import Embedder, load_tokenizer, load_base_model, POOLING_MODES


def normalize_lang(code):
    """Normalize language code to base language (e.g. pt_BR -> pt)."""
    try:
        return langcodes.Language.get(code).language
    except:
        return code


# All language pairs including English (430 total)
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

# Curated high-resource language pairs (69 pairs)
# Chinese, Spanish, Hindi, Portuguese, Bengali, Russian, Japanese, Punjabi, Vietnamese,
# Turkish, Arabic, Marathi, Telugu, Korean, Tamil, Urdu, German, Indonesian, French,
# Javanese, Persian, Italian, Hausa, Gujarati, Bhojpuri
CURATED_LANG_PAIRS = [
    # Arabic variants
    "acm_Arab-eng_Latn", "aeb_Arab-eng_Latn", "apc_Arab-eng_Latn", "ara_Arab-eng_Latn",
    "arb_Arab-eng_Latn", "arq_Arab-eng_Latn", "ars_Arab-eng_Latn", "ary_Arab-eng_Latn",
    "arz_Arab-eng_Latn", "eng_Latn-aeb_Arab", "eng_Latn-arb_Arab", "eng_Latn-ars_Arab",
    "eng_Latn-arz_Arab",
    # Bengali
    "ben_Beng-eng_Latn", "eng_Latn-ben_Beng",
    # Bhojpuri
    "bho_Deva-eng_Latn", "eng_Latn-bho_Deva",
    # Chinese variants
    "cmn_Hani-eng_Latn", "eng_Latn-cmn_Hani", "eng_Latn-lzh_Hani", "eng_Latn-wuu_Hani",
    "eng_Latn-yue_Hani", "eng_Latn-zho_Hani", "eng_Latn-zho_Hans", "eng_Latn-zho_Hant",
    "eng_Latn-zho_Zyyy",
    # German
    "deu_Latn-eng_Latn", "eng_Latn-deu_Latn",
    # Persian
    "fas_Arab-eng_Latn", "eng_Latn-fas_Arab", "eng_Latn-prs_Arab",
    # French
    "fra_Latn-eng_Latn", "eng_Latn-fra_Latn",
    # Gujarati
    "eng_Latn-guj_Gujr",
    # Hausa
    "eng_Latn-hau_Latn",
    # Hindi
    "hin_Deva-eng_Latn", "eng_Latn-hin_Deva", "eng_Latn-hin_Latn",
    # Indonesian
    "ind_Latn-eng_Latn", "eng_Latn-ind_Latn",
    # Italian
    "ita_Latn-eng_Latn", "eng_Latn-ita_Latn",
    # Javanese
    "eng_Latn-jav_Latn",
    # Japanese
    "jpn_Jpan-eng_Latn", "eng_Latn-jpn_Jpan",
    # Korean
    "kor_Hang-eng_Latn", "kor_Zyyy-eng_Latn", "eng_Latn-kor_Hang", "eng_Latn-kor_Zyyy",
    # Marathi
    "eng_Latn-mar_Deva", "eng_Latn-mar_Latn",
    # Punjabi
    "eng_Latn-pan_Guru", "eng_Latn-pan_Latn", "eng_Latn-pnb_Arab",
    # Portuguese
    "por_Latn-eng_Latn", "eng_Latn-por_Latn",
    # Russian
    "rus_Cyrl-eng_Latn", "eng_Latn-rus_Cyrl",
    # Spanish
    "spa_Latn-eng_Latn", "eng_Latn-spa_Latn",
    # Tamil
    "eng_Latn-tam_Taml",
    # Telugu
    "eng_Latn-tel_Latn", "eng_Latn-tel_Telu",
    # Turkish
    "tur_Latn-eng_Latn", "eng_Latn-tur_Latn",
    # Urdu
    "eng_Latn-urd_Arab", "eng_Latn-urd_Latn",
    # Vietnamese
    "vie_Latn-eng_Latn", "eng_Latn-vie_Latn",
]

# Default validation pairs: 10 diverse languages, both directions (20 pairs)
# Covers: CJK (ja, zh, ko), Arabic script (ar), Devanagari (hi), Cyrillic (ru),
# Latin (es, de, vi, tr) - representing Romance, Germanic, Tonal, Turkic
VALIDATION_PAIRS = [
    # Japanese
    "jpn_Jpan-eng_Latn", "eng_Latn-jpn_Jpan",
    # Chinese (Mandarin)
    "cmn_Hani-eng_Latn", "eng_Latn-cmn_Hani",
    # Arabic (Standard)
    "arb_Arab-eng_Latn", "eng_Latn-arb_Arab",
    # Hindi
    "hin_Deva-eng_Latn", "eng_Latn-hin_Deva",
    # Russian
    "rus_Cyrl-eng_Latn", "eng_Latn-rus_Cyrl",
    # Spanish
    "spa_Latn-eng_Latn", "eng_Latn-spa_Latn",
    # German
    "deu_Latn-eng_Latn", "eng_Latn-deu_Latn",
    # Korean
    "kor_Hang-eng_Latn", "eng_Latn-kor_Hang",
    # Vietnamese
    "vie_Latn-eng_Latn", "eng_Latn-vie_Latn",
    # Turkish
    "tur_Latn-eng_Latn", "eng_Latn-tur_Latn",
]


def contrastive_loss(a, b, temp=0.05):
    """Contrastive loss for normalized embeddings."""
    logits = (a @ b.T) / temp
    labels = torch.arange(a.size(0), device=a.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_j) / 2


def estimate_tokens(text):
    """Fast token estimate: ~4 chars per token for LLaMA."""
    return len(text) // 4 + 1


def parse_num(s):
    """Parse human-readable numbers like 1B, 500M, 100K."""
    s = s.strip().upper()
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune multilingual embeddings")

    # Model
    parser.add_argument("--model", type=str, default="MaLA-LM/emma-500-llama3-8b-bi",
                        help="Base model ID")
    parser.add_argument("--out-dim", type=int, default=768,
                        help="Output embedding dimension")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Hidden layer to extract embeddings from")
    parser.add_argument("--pooling", type=str, default="mean", choices=POOLING_MODES,
                        help="Pooling strategy: mean, last, or attention")
    parser.add_argument("--mlp-head", action="store_true",
                        help="Use MLP projection head (Linear->ReLU->Linear) instead of single Linear")
    parser.add_argument("--mlp-hidden", type=int, default=2048,
                        help="Hidden dimension for MLP head")

    # Devices
    parser.add_argument("--train-device", type=str, default="cuda:0",
                        help="Device for training")
    parser.add_argument("--val-device", type=str, default="cuda:0",
                        help="Device for validation (set same as train-device for single GPU)")

    # Training
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Temperature for contrastive loss (lower = sharper)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of training steps (overrides --total-tokens)")
    parser.add_argument("--total-tokens", type=parse_num, default="10M",
                        help="Total tokens to train on (e.g. 1M, 500K, 1B)")
    parser.add_argument("--max-batch-tokens", type=int, default=3072,
                        help="Max tokens per batch")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--checkpoint-steps", type=int, default=100,
                        help="Save checkpoint every N steps (when using --steps)")
    parser.add_argument("--checkpoint-tokens", type=parse_num, default="1M",
                        help="Save checkpoint every N tokens (e.g. 100K, 1M)")
    parser.add_argument("--val-count", type=int, default=2,
                        help="Number of validations during training (evenly spaced + final)")

    # Data
    parser.add_argument("--lang-set", type=str, default="curated", choices=["all", "curated"],
                        help="Language pair set: 'all' (430 pairs) or 'curated' (69 high-resource pairs)")
    parser.add_argument("--num-langs", type=int, default=69,
                        help="Sample N language pairs from the set (default: use all in set)")
    parser.add_argument("--train-pairs", type=str, nargs="+", default=None,
                        help="Explicit language pairs for training (overrides --lang-set and --num-langs)")
    parser.add_argument("--val-pairs", type=str, nargs="+", default=VALIDATION_PAIRS,
                        help="Language pairs for validation (budget shared between them)")
    parser.add_argument("--val-size", type=int, default=1000,
                        help="Total number of validation sentence pairs (shared across all val-pairs)")
    parser.add_argument("--val-batch-tokens", type=int, default=512,
                        help="Max tokens per validation batch (default: same as --max-batch-tokens)")
    parser.add_argument("--shuffle-buffer", type=int, default=1000,
                        help="Shuffle buffer size (lower = faster startup)")
    parser.add_argument("--adaptive-sampling", action="store_true",
                        help="Adjust sampling weights based on validation loss (oversample hard pairs)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (show per-language and per-pair details)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars")

    return parser.parse_args()


def main():
    args = parse_args()

    # Print settings
    print("=" * 60)
    print("Settings:")
    print(f"  Model:        {args.model}")
    print(f"  Layer:        {args.layer}")
    print(f"  Output dim:   {args.out_dim}")
    print(f"  Pooling:      {args.pooling}")
    print(f"  MLP head:     {args.mlp_head}" + (f" (hidden={args.mlp_hidden})" if args.mlp_head else ""))
    print(f"  Temperature:  {args.temperature}")
    print(f"  LR:           {args.lr}")
    if args.steps:
        print(f"  Steps:        {args.steps:,}")
    else:
        print(f"  Total tokens: {args.total_tokens:,}")
    print(f"  Batch tokens: {args.max_batch_tokens}")
    print(f"  Max length:   {args.max_length}")
    if args.train_pairs:
        print(f"  Train pairs:  {len(args.train_pairs)} explicit pairs")
    else:
        print(f"  Lang set:     {args.lang_set} ({args.num_langs} pairs)")
    print(f"  Val pairs:    {len(args.val_pairs)} pairs, {args.val_size} examples")
    print(f"  Val count:    {args.val_count}")
    print(f"  Adaptive:     {args.adaptive_sampling}")
    print(f"  Devices:      train={args.train_device}, val={args.val_device}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Seed:         {args.seed}")
    print("=" * 60)

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = load_tokenizer(args.model)

    def tokenize(texts, max_length, device):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {k: v.to(device) for k, v in batch.items()}

    # Load base model for training
    print(f"Loading base model on {args.train_device}...")
    base = load_base_model(args.model, args.train_device)

    # Load base model for validation (separate copy if different device)
    if args.val_device != args.train_device:
        torch.cuda.empty_cache()
        print(f"Loading validation model on {args.val_device}...")
        val_base = load_base_model(args.model, args.val_device)
    else:
        val_base = base

    # Create embedder(s)
    embedder = Embedder(base, out_dim=args.out_dim, layer=args.layer, pooling=args.pooling,
                        mlp_head=args.mlp_head, mlp_hidden=args.mlp_hidden).to(dtype=torch.bfloat16, device=args.train_device).train()
    if args.val_device != args.train_device:
        val_embedder = Embedder(val_base, out_dim=args.out_dim, layer=args.layer, pooling=args.pooling,
                                mlp_head=args.mlp_head, mlp_hidden=args.mlp_hidden).to(dtype=torch.bfloat16, device=args.val_device).eval()
    else:
        val_embedder = None  # Use embedder for validation when on same device

    # Optimizer
    opt = torch.optim.AdamW(embedder.parameters(), lr=args.lr)

    # Select language pairs
    if args.train_pairs:
        # Explicit training pairs override lang-set and num-langs
        selected_pairs = args.train_pairs
        print(f"Using {len(selected_pairs)} explicit training pairs")
        if args.verbose:
            for lp in selected_pairs:
                print(f"  {lp}")
    else:
        base_pairs = CURATED_LANG_PAIRS if args.lang_set == "curated" else LANG_PAIRS
        set_name = "curated" if args.lang_set == "curated" else "all"

        if args.num_langs is not None:
            selected_pairs = random.sample(base_pairs, min(args.num_langs, len(base_pairs)))
            print(f"Sampled {len(selected_pairs)} from {set_name} set ({len(base_pairs)} pairs)")
            if args.verbose:
                for lp in selected_pairs:
                    print(f"  {lp}")
        else:
            selected_pairs = base_pairs
            print(f"Using {set_name} set: {len(selected_pairs)} language pairs")

    # Load training datasets
    print(f"Loading {len(selected_pairs)} training datasets" + (" (adaptive)" if args.adaptive_sampling else "") + "...")
    if args.adaptive_sampling:
        # Keep individual iterators for weighted sampling
        pair_iterators = {}
        for lp in tqdm(selected_pairs, desc="Loading language pairs", disable=args.no_progress):
            ds = load_dataset("MaLA-LM/FineOPUS-ReLID", data_dir=lp, split="train", streaming=True)
            ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            pair_iterators[lp] = iter(ds)
        sampling_weights = {lp: 1.0 for lp in selected_pairs}
        data_iter = None  # Will sample manually in get_batch
    else:
        datasets = [
            load_dataset("MaLA-LM/FineOPUS-ReLID", data_dir=lp, split="train", streaming=True)
            for lp in tqdm(selected_pairs, desc="Loading language pairs", disable=args.no_progress)
        ]
        dataset = interleave_datasets(datasets)
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        pair_iterators = None
        sampling_weights = None

    # Load validation set from multiple pairs (budget shared)
    val_batch_tokens = args.val_batch_tokens if args.val_batch_tokens is not None else args.max_batch_tokens
    val_pairs = args.val_pairs
    per_pair_size = args.val_size // len(val_pairs)
    print(f"Creating validation set: {len(val_pairs)} pairs, {args.val_size} total examples...")

    # Load and batch per language pair (for per-pair loss computation)
    val_batches_by_pair = {}
    total_val_batches = 0
    val_pair_info = []  # Collect for sorted output
    for val_pair in val_pairs:
        val_dataset = load_dataset("MaLA-LM/FineOPUS-ReLID", data_dir=val_pair, split="train", streaming=True)
        val_iter = iter(val_dataset)
        val_left, val_right, val_tokens = [], [], []
        while len(val_left) < per_pair_size:
            sample = next(val_iter)
            src, tgt = sample["source_text"], sample["target_text"]
            src_tok, tgt_tok = estimate_tokens(src), estimate_tokens(tgt)
            if src_tok > args.max_length or tgt_tok > args.max_length:
                continue
            val_left.append(src)
            val_right.append(tgt)
            val_tokens.append(max(src_tok, tgt_tok))

        # Batch this pair's data
        batches = []
        batch_left, batch_right, batch_tok = [], [], 0
        for src, tgt, tok in zip(val_left, val_right, val_tokens):
            if batch_tok + tok > val_batch_tokens and batch_left:
                batches.append((
                    tokenize(batch_left, args.max_length, args.val_device),
                    tokenize(batch_right, args.max_length, args.val_device)
                ))
                batch_left, batch_right, batch_tok = [], [], 0
            batch_left.append(src)
            batch_right.append(tgt)
            batch_tok += tok
        if batch_left:
            batches.append((
                tokenize(batch_left, args.max_length, args.val_device),
                tokenize(batch_right, args.max_length, args.val_device)
            ))
        val_batches_by_pair[val_pair] = batches
        total_val_batches += len(batches)
        src_lang, tgt_lang = val_pair.split("-")
        short_pair = f"{normalize_lang(src_lang)}-{normalize_lang(tgt_lang)}"
        val_pair_info.append((short_pair, len(val_left), len(batches)))
    # Print sorted verbose output
    if args.verbose:
        for short_pair, n_pairs, n_batches in sorted(val_pair_info):
            print(f"  {short_pair}: {n_pairs} pairs, {n_batches} batches")
    print(f"Validation ready: {total_val_batches} batches across {len(val_pairs)} pairs")

    # Create training iterator (only for non-adaptive mode)
    if not args.adaptive_sampling:
        data_iter = iter(dataset)

    # Async validation state
    val_thread = None
    val_loss_result = [None]  # overall loss
    val_loss_by_pair = {}  # per-pair loss

    def run_validation():
        if val_embedder is not None:
            # Different devices: copy weights and run on validation model
            state = {k: v.to(args.val_device) for k, v in embedder.proj.state_dict().items()}
            val_embedder.proj.load_state_dict(state)
            model = val_embedder
        else:
            # Same device: use embedder directly in eval mode
            embedder.eval()
            model = embedder

        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for val_pair, batches in val_batches_by_pair.items():
                pair_loss = 0.0
                for val_batch_l, val_batch_r in batches:
                    val_a = model(**val_batch_l)
                    val_b = model(**val_batch_r)
                    pair_loss += contrastive_loss(val_a, val_b, args.temperature).item()
                val_loss_by_pair[val_pair] = pair_loss / len(batches) if batches else 0.0
                total_loss += pair_loss
                total_batches += len(batches)

        if val_embedder is None:
            embedder.train()

        val_loss_result[0] = total_loss / total_batches if total_batches > 0 else 0.0

    # Statistics
    lang_tokens = Counter()  # tokens per language
    lang_sents = Counter()  # sentences per language
    pair_examples = Counter()  # examples per language pair
    total_examples = [0]
    total_tokens_seen = [0]
    skipped_too_long = [0]
    stats_lock = threading.Lock()

    def sample_from_pair():
        """Sample one example from pair iterators based on weights."""
        pairs = list(sampling_weights.keys())
        weights = [sampling_weights[p] for p in pairs]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen_pair = random.choices(pairs, weights=probs, k=1)[0]
        return next(pair_iterators[chosen_pair])

    def get_batch():
        left, right = [], []
        total_tokens = 0
        batch_lang_tokens = Counter()
        batch_lang_sents = Counter()
        batch_pair_examples = Counter()
        batch_skipped = 0
        batch_examples = 0
        while total_tokens < args.max_batch_tokens:
            if args.adaptive_sampling:
                sample = sample_from_pair()
            else:
                sample = next(data_iter)
            src, tgt = sample["source_text"], sample["target_text"]
            src_tok, tgt_tok = estimate_tokens(src), estimate_tokens(tgt)
            if src_tok > args.max_length or tgt_tok > args.max_length:
                batch_skipped += 1
                continue
            src_lang = normalize_lang(sample['orig_src_lang'])
            tgt_lang = normalize_lang(sample['orig_tgt_lang'])
            batch_lang_tokens[src_lang] += src_tok
            batch_lang_tokens[tgt_lang] += tgt_tok
            batch_lang_sents[src_lang] += 1
            batch_lang_sents[tgt_lang] += 1
            batch_pair_examples[f"{src_lang}-{tgt_lang}"] += 1
            total_tokens += max(src_tok, tgt_tok)
            batch_examples += 1
            left.append(src)
            right.append(tgt)
        with stats_lock:
            lang_tokens.update(batch_lang_tokens)
            lang_sents.update(batch_lang_sents)
            pair_examples.update(batch_pair_examples)
            total_examples[0] += batch_examples
            total_tokens_seen[0] += sum(batch_lang_tokens.values())
            skipped_too_long[0] += batch_skipped
        return left, right, total_tokens

    def print_stats(top_n=5):
        with stats_lock:
            n_examples = total_examples[0]
            n_langs = len(lang_tokens)
            n_toks = total_tokens_seen[0]
            avg_tok = n_toks / (n_examples * 2) if n_examples > 0 else 0
            print(f"Stats: {n_examples:,} examples, {n_langs} langs, {n_toks:,} toks, {avg_tok:.1f} avg tok/sent", flush=True)
            if args.verbose:
                # Verbose: full per-language stats
                print("  Tokens per language:", flush=True)
                for lang, toks in lang_tokens.most_common(top_n * 2):
                    sents = lang_sents[lang]
                    avg = toks / sents if sents > 0 else 0
                    print(f"    {lang}: {toks:,} ({100*toks/n_toks:.1f}%, {avg:.1f} avg)", flush=True)
                print("  Examples per language pair:", flush=True)
                for pair, count in pair_examples.most_common(top_n * 2):
                    print(f"    {pair}: {count:,} ({100*count/n_examples:.1f}%)", flush=True)
            else:
                # Compact: top languages on one line
                top_langs = [f"{lang}:{toks//1000}k" for lang, toks in lang_tokens.most_common(top_n)]
                print(f"  Top langs: {', '.join(top_langs)}", flush=True)
            # Validation loss per pair (sorted by normalized pair name)
            if val_loss_by_pair:
                def sort_key(item):
                    src, tgt = item[0].split("-")
                    return f"{normalize_lang(src)}-{normalize_lang(tgt)}"
                sorted_losses = sorted(val_loss_by_pair.items(), key=sort_key)
                if args.verbose:
                    print("  Validation loss per pair:", flush=True)
                    for val_pair, loss in sorted_losses:
                        src_lang, tgt_lang = val_pair.split("-")
                        short_pair = f"{normalize_lang(src_lang)}-{normalize_lang(tgt_lang)}"
                        weight_str = ""
                        if args.adaptive_sampling and val_pair in sampling_weights:
                            weight_str = f" (weight: {sampling_weights[val_pair]:.2f})"
                        print(f"    {short_pair}: {loss:.4f}{weight_str}", flush=True)
                else:
                    # Compact: multiple per line
                    loss_items = []
                    for val_pair, loss in sorted_losses:
                        src_lang, tgt_lang = val_pair.split("-")
                        short_pair = f"{normalize_lang(src_lang)}-{normalize_lang(tgt_lang)}"
                        loss_items.append(f"{short_pair}:{loss:.3f}")
                    for i in range(0, len(loss_items), 5):
                        print(f"  Val: {', '.join(loss_items[i:i+5])}", flush=True)

    def update_sampling_weights():
        """Update sampling weights based on validation loss (higher loss = higher weight)."""
        if not args.adaptive_sampling or not val_loss_by_pair:
            return
        # Update weights for pairs that are in both training and validation
        validated_weights = []
        for val_pair, loss in val_loss_by_pair.items():
            if val_pair in sampling_weights:
                weight = max(0.1, min(10.0, loss))
                sampling_weights[val_pair] = weight
                validated_weights.append(weight)
        # Set non-validated pairs to mean weight of validated pairs
        mean_weight = sum(validated_weights) / len(validated_weights) if validated_weights else 1.0
        for pair in sampling_weights:
            if pair not in val_loss_by_pair:
                sampling_weights[pair] = mean_weight
        if args.verbose:
            print(f"  Updated sampling weights (non-validated: {mean_weight:.2f}):", flush=True)
            total_w = sum(sampling_weights.values())
            # Sort by normalized pair name
            validated_pairs = [(p, sampling_weights[p]) for p in sampling_weights if p in val_loss_by_pair]
            validated_pairs.sort(key=lambda x: f"{normalize_lang(x[0].split('-')[0])}-{normalize_lang(x[0].split('-')[1])}")
            for pair, weight in validated_pairs:
                src_lang, tgt_lang = pair.split("-")
                short_pair = f"{normalize_lang(src_lang)}-{normalize_lang(tgt_lang)}"
                prob = weight / total_w * 100
                print(f"    {short_pair}: {weight:.2f} ({prob:.1f}%)", flush=True)
        else:
            print(f"  Adaptive weights updated (base: {mean_weight:.2f})", flush=True)

    # Prefetch batches in background
    batch_queue = Queue(maxsize=4)
    stop_prefetch = threading.Event()

    def prefetch_worker():
        while not stop_prefetch.is_set():
            try:
                left, right, batch_tokens = get_batch()
                batch_l = tokenize(left, args.max_length, args.train_device)
                batch_r = tokenize(right, args.max_length, args.train_device)
                # Use timeout to allow checking stop_prefetch
                while not stop_prefetch.is_set():
                    try:
                        batch_queue.put((batch_l, batch_r, batch_tokens), timeout=0.1)
                        break
                    except:
                        pass
            except StopIteration:
                break

    prefetch_thread = threading.Thread(target=prefetch_worker)
    prefetch_thread.start()

    # Training loop
    tokens_processed = 0
    step = 0
    val_idx = 0  # Next validation index
    use_steps = args.steps is not None

    # Calculate validation points (evenly spaced)
    total_budget = args.steps if use_steps else args.total_tokens
    val_points = [total_budget * (i + 1) // args.val_count for i in range(args.val_count)]

    if use_steps:
        pbar = tqdm(total=args.steps, desc="Training", unit="step", disable=args.no_progress)
    else:
        pbar = tqdm(total=args.total_tokens, desc="Training", unit="tok", disable=args.no_progress)

    def should_continue():
        if use_steps:
            return step < args.steps
        return tokens_processed < args.total_tokens

    def should_validate():
        nonlocal val_idx
        if val_idx >= len(val_points):
            return False
        progress = step if use_steps else tokens_processed
        if progress >= val_points[val_idx]:
            val_idx += 1
            return True
        return False

    while should_continue():
        batch_l, batch_r, batch_tokens = batch_queue.get()

        a = embedder(**batch_l)
        b = embedder(**batch_r)

        loss = contrastive_loss(a, b, args.temperature)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tokens_processed += batch_tokens
        step += 1
        pbar.update(1 if use_steps else batch_tokens)

        # Update progress bar
        pbar.set_postfix(
            train=f"{loss.detach().item():.4f}",
            val=f"{val_loss_result[0]:.4f}" if val_loss_result[0] else "..."
        )

        if should_validate():
            # Wait for previous validation
            if val_thread is not None:
                val_thread.join()

            # Launch validation
            val_thread = threading.Thread(target=run_validation)
            val_thread.start()
            val_thread.join()

            # Update progress bar with validation result
            pbar.set_postfix(train=f"{loss.detach().item():.4f}", val=f"{val_loss_result[0]:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir, f"embedder_{tokens_processed // 1000}k.pt")
            torch.save(embedder.proj.state_dict(), ckpt_path)
            if args.pooling == "attention":
                torch.save({'attn_query': embedder.attn_query.data},
                           ckpt_path.replace('.pt', '_attn.pt'))

            # Print checkpoint info
            pbar.clear()
            print(f"\nCheckpoint: {tokens_processed:,} tokens, train={loss.detach().item():.4f}, val={val_loss_result[0]:.4f}", flush=True)
            print_stats()
            update_sampling_weights()
            pbar.refresh()

    pbar.close()

    # Cleanup
    stop_prefetch.set()
    prefetch_thread.join()
    if val_thread is not None:
        val_thread.join()

    # Print final stats
    print(f"\nTraining complete: {step:,} steps, {tokens_processed:,} tokens, {total_examples[0]:,} examples, {skipped_too_long[0]:,} skipped (exceeded max_length)")

    # Save final model
    final_path = os.path.join(args.output_dir, "embedder.pt")
    torch.save(embedder.proj.state_dict(), final_path)
    if args.pooling == "attention":
        torch.save({'attn_query': embedder.attn_query.data},
                   final_path.replace('.pt', '_attn.pt'))
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
