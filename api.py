import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

print("INIT SERVER...")

# Configuración
ruta_modelo = "multilangs_model"

# Detección de Hardware
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Hardware: {device}")

# Cargar configuración y tokenizador
config = PeftConfig.from_pretrained(ruta_modelo)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Cargar Modelo Base
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)

# Cargar LoRA 
model = PeftModel.from_pretrained(model_base, ruta_modelo)
model = model.to(device)
model.eval()

app = FastAPI(title="API Traductor NLLB-LoRA Multilingüe")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapeo de códigos de idioma
CODIGOS = {
    "es": "spa_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans"
}

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "es"  # es, it, ja, zh
    target_lang: str = "it"  # es, it, ja, zh

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Validar idiomas
        if request.source_lang not in CODIGOS or request.target_lang not in CODIGOS:
            raise HTTPException(status_code=400, detail="Idioma no soportado. Usa: es, it, ja, zh")
        
        src_code = CODIGOS[request.source_lang]
        tgt_code = CODIGOS[request.target_lang]
        
        # Configurar idiomas
        tokenizer.src_lang = src_code
        
        # Procesar entrada
        inputs = tokenizer(request.text, return_tensors="pt").to(device)
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
        
        with torch.no_grad():
            # Generar traducción
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=100,
                num_beams=5,
                early_stopping=True
            )
            
            translation_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            # Extraer Atención
            tokenizer.tgt_lang = tgt_code
            target_inputs = tokenizer(
                text_target=translation_text, 
                return_tensors="pt",
                add_special_tokens=True
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=target_inputs["input_ids"],
                output_attentions=True 
            )
            
            layer_2_attention = outputs.cross_attentions[2][0]
            avg_attention = layer_2_attention.mean(dim=0)
            attention_matrix = avg_attention.cpu().numpy().tolist()

            # Preparar tokens
            src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            src_tokens = [t.replace('▁', '') for t in src_tokens]
            
            tgt_tokens = tokenizer.convert_ids_to_tokens(target_inputs["input_ids"][0])
            tgt_tokens = [t.replace('▁', '') for t in tgt_tokens]

        return {
            "original": request.text,
            "translation": translation_text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "device": device,
            "attention": {
                "matrix": attention_matrix,
                "src_tokens": src_tokens,
                "tgt_tokens": tgt_tokens
            }
        }

    except Exception as e:
        print(f"Error detallado: {e}") 
        raise HTTPException(status_code=500, detail=str(e))