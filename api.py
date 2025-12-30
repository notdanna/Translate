import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import gc

print("INIT SERVER...")

# Configuracion
ruta_modelo = "multilangs_model"

# Deteccion de Hardware
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Hardware: {device}")

# Cargar configuracion
config = PeftConfig.from_pretrained(ruta_modelo)

# Cargar Modelo Base
model_base = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)

# Cargar LoRA 

model = PeftModel.from_pretrained(model_base, ruta_modelo)
#model = model_base
model = model.to(device)
model.eval()

app = FastAPI(title="API Traductor NLLB-LoRA Multilingual")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapeo de codigos de idioma
CODIGOS = {
    "es": "spa_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans"
}

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "es"
    target_lang: str = "it"

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Validar idiomas
        if request.source_lang not in CODIGOS or request.target_lang not in CODIGOS:
            raise HTTPException(status_code=400, detail="Idioma no soportado")
        
        src_code = CODIGOS[request.source_lang]
        tgt_code = CODIGOS[request.target_lang]
        
        print(f"\nNueva traduccion: {request.source_lang} -> {request.target_lang}")
        print(f"Texto: {request.text}")
        print(f"Codigos: {src_code} -> {tgt_code}")
        
        # TOKENIZER CON IDIOMA ORIGEN
        tokenizer = NllbTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.src_lang = src_code
        
        # Tokenizar entrada
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200
        ).to(device)
        
        # CONFIGURAR IDIOMA DESTINO ANTES DE GENERAR
        tokenizer.tgt_lang = tgt_code
        
        # Obtener el token de idioma destino
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)
        
        print(f"Source lang: {src_code}")
        print(f"Target lang: {tgt_code}")
        print(f"Target lang ID: {tgt_lang_id}")
        print(f"Input tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        
        # GENERAR CON NO_REPEAT_NGRAM ACTIVADO
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=200,
                num_beams=5,
                
                # CRITICO: PREVENIR REPETICIONES
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False,
                use_cache=True,
            )
            
            print(f"Generated shape: {generated_tokens.shape}")
            print(f"Generated tokens: {tokenizer.convert_ids_to_tokens(generated_tokens[0][:20])}")
            
            # Decodificar
            translation_text = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            print(f"Traduccion: {translation_text}")
            
            # EXTRAER ATENCION
            tokenizer_attn = NllbTokenizer.from_pretrained(config.base_model_name_or_path)
            tokenizer_attn.src_lang = src_code
            tokenizer_attn.tgt_lang = tgt_code
            
            target_inputs = tokenizer_attn(
                translation_text,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=target_inputs["input_ids"],
                output_attentions=True,
                use_cache=False
            )
            
            layer_2_attention = outputs.cross_attentions[2][0]
            avg_attention = layer_2_attention.mean(dim=0)
            attention_matrix = avg_attention.cpu().numpy().tolist()

            src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            src_tokens = [t.replace('▁', '') for t in src_tokens]
            
            tgt_tokens = tokenizer.convert_ids_to_tokens(target_inputs["input_ids"][0])
            tgt_tokens = [t.replace('▁', '') for t in tgt_tokens]

        # LIMPIEZA DE MEMORIA
        del inputs, target_inputs, outputs, generated_tokens
        del tokenizer, tokenizer_attn
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.empty_cache()
        
        gc.collect()
        
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
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Limpieza en error
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        
        raise HTTPException(status_code=500, detail=str(e))