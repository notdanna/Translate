import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import seaborn as sns
import numpy as np
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Al inicio del script, después de los imports
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False




# --- CONFIGURACIÓN ---
ruta_modelo = "multilangs_model"
print(f"Cargando modelo...")

config = PeftConfig.from_pretrained(ruta_modelo)
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    dtype=torch.float32, 
    low_cpu_mem_usage=True,
    attn_implementation="eager" 
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# model = PeftModel.from_pretrained(model_base, ruta_modelo)

if torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"


model = model_base  # USA EL MODELO BASE SIN LORA
model = model.to(device)
model.eval()



# Mapeo de códigos de idioma
CODIGOS = {
    "es": "spa_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans"
}

# --- FUNCIÓN DE TRADUCCIÓN ---
def obtener_atencion_capa_2(texto, src_lang="es", tgt_lang="it"):
    src_code = CODIGOS[src_lang]
    tgt_code = CODIGOS[tgt_lang]
    
    tokenizer.src_lang = src_code
    inputs = tokenizer(texto, return_tensors="pt").to(device)
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, forced_bos_token_id=forced_bos_id, max_length=50
        )
        traduccion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        tokenizer.tgt_lang = tgt_code
        target_inputs = tokenizer(text_target=traduccion, return_tensors="pt", add_special_tokens=True).to(device)
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=target_inputs["input_ids"],
            output_attentions=True
        )
        
        atencion_capa_2 = outputs.cross_attentions[2][0].mean(dim=0).cpu().numpy()
        
        return traduccion, atencion_capa_2, inputs, target_inputs

# --- GRAFICAR CAPA ÚNICA ---
def graficar_capa_2(matrix, inputs, target_inputs, src_lang="es", tgt_lang="it"):
    src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tgt_tokens = tokenizer.convert_ids_to_tokens(target_inputs["input_ids"][0])
    
    src_tokens = [t.replace('▁', '') for t in src_tokens]
    tgt_tokens = [t.replace('▁', '') for t in tgt_tokens]
    
    matrix_clean = matrix[1:-1, 1:-1]
    src_clean = src_tokens[1:-1]
    tgt_clean = tgt_tokens[1:-1]

    mins = matrix_clean.min(axis=1, keepdims=True)
    maxs = matrix_clean.max(axis=1, keepdims=True)
    norm_matrix = (matrix_clean - mins) / (maxs - mins + 1e-10)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        norm_matrix, 
        xticklabels=src_clean, 
        yticklabels=tgt_clean, 
        cmap="Blues", 
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='lightgray'
    )
    plt.title(f"Alineación de Traducción (Capa 2) - {src_lang.upper()} → {tgt_lang.upper()}")
    plt.xlabel(src_lang.upper())
    plt.ylabel(tgt_lang.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# --- EJECUCIÓN ---
frase = "me quiero matar antes de hoy"
src = "es"  # Cambiar según necesites: es, it, ja, zh
tgt = "ja"  # Cambiar según necesites: es, it, ja, zh

print(f"Entrada ({src}): {frase}")

trad, matriz, inp, tgt_inp = obtener_atencion_capa_2(frase, src, tgt)
print(f"Salida ({tgt}):  {trad}")

graficar_capa_2(matriz, inp, tgt_inp, src, tgt)