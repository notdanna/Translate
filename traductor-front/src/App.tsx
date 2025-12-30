import { useState, useEffect } from 'react';
import { Mic, MicOff, Languages, ArrowRightLeft, Loader2, Copy, Check, BrainCircuit, Volume2 } from 'lucide-react';
import useSpeechToText from './useSpeechToText';

const speakText = (text: string, lang: string) => {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    
    const langMap: { [key: string]: string } = {
      'es': 'es-ES',
      'it': 'it-IT',
      'ja': 'ja-JP',
      'zh': 'zh-CN'
    };
    
    utterance.lang = langMap[lang] || 'es-ES';
    utterance.rate = 0.9;
    utterance.pitch = 1;
    
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }
};

const API_URL = "http://127.0.0.1:8000/translate";

interface AttentionData {
  matrix: number[][];
  src_tokens: string[];
  tgt_tokens: string[];
}

const LANGUAGES = {
  es: { name: 'Español', flag: 'ES' },
  it: { name: 'Italiano', flag: 'IT' },
  ja: { name: 'Japonés', flag: 'JP' },
  zh: { name: 'Chino', flag: 'CN' }
};

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [attentionData, setAttentionData] = useState<AttentionData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [sourceLang, setSourceLang] = useState<keyof typeof LANGUAGES>('es');
  const [targetLang, setTargetLang] = useState<keyof typeof LANGUAGES>('it');

  const { isListening, transcript, startListening, stopListening, hasSupport } = useSpeechToText(sourceLang);

  useEffect(() => {
    if (transcript) setInputText(transcript);
  }, [transcript]);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setOutputText('');
    setAttentionData(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          source_lang: sourceLang,
          target_lang: targetLang
        })
      });

      if (!response.ok) throw new Error("Error en la red");
      
      const data = await response.json();
      setOutputText(data.translation);
      
      if (data.attention) {
        setAttentionData(data.attention);
      }

    } catch (error) {
      console.error("Error:", error);
      setOutputText("Error al conectar con la API.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const swapLanguages = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    setInputText(outputText);
    setOutputText('');
    setAttentionData(null);
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-50 overflow-hidden">
      
      {/* Header Compacto */}
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-blue-600 rounded-lg text-white">
            <Languages size={20} />
          </div>
          <h1 className="text-xl font-bold text-slate-800 tracking-tight">Traductor ML - Transformers</h1>
        </div>
      </header>

      {/* Contenedor Principal Side-by-Side */}
      <main className="flex-1 flex overflow-hidden p-4 gap-4">
        
        {/* Panel Izquierdo: Chat / Traducción */}
        <div className="w-full max-w-[700px] flex flex-col gap-4 shrink-0">
          
          {/* Caja de Entrada */}
          <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
            <div className="p-3 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
              <select 
                value={sourceLang}
                onChange={(e) => setSourceLang(e.target.value as keyof typeof LANGUAGES)}
                className="font-bold text-slate-700 bg-white border border-slate-200 px-2 py-1 rounded-lg text-xs outline-none cursor-pointer"
              >
                {Object.entries(LANGUAGES).map(([code, lang]) => (
                  <option key={code} value={code}>{lang.flag} {lang.name}</option>
                ))}
              </select>
              {hasSupport && (
                <button 
                  onClick={isListening ? stopListening : startListening}
                  className={`p-1.5 rounded-full transition-all ${isListening ? 'bg-red-100 text-red-600 animate-pulse' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}
                >
                  {isListening ? <MicOff size={16}/> : <Mic size={16}/>}
                </button>
              )}
            </div>
            <textarea
              className="flex-1 p-4 w-full bg-transparent resize-none outline-none text-lg text-slate-800 placeholder:text-slate-300"
              placeholder="Escribe algo aquí..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleTranslate(); }}}
            />
            <div className="p-3 border-t border-slate-50 flex justify-end">
                <button 
                    onClick={handleTranslate} 
                    disabled={isLoading || !inputText.trim()}
                    className="bg-blue-600 text-white px-4 py-2 rounded-xl shadow-md hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2 text-sm font-bold transition-all"
                >
                    {isLoading ? <Loader2 size={16} className="animate-spin"/> : <ArrowRightLeft size={16} />}
                    <span>Traducir</span>
                </button>
            </div>
          </div>

          {/* Botón Swap entre cajas */}
          <div className="flex justify-center -my-2 z-10">
            <button onClick={swapLanguages} className="bg-white border border-slate-200 p-2 rounded-full shadow-sm hover:bg-slate-50 text-slate-500 transition-transform active:scale-90">
              <ArrowRightLeft size={16} className="rotate-90" />
            </button>
          </div>

          {/* Caja de Salida */}
          <div className="flex-1 bg-blue-50/30 rounded-2xl shadow-sm border border-blue-100 flex flex-col overflow-hidden">
            <div className="p-3 border-b border-blue-100 flex justify-between items-center bg-blue-50/50">
              <select 
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value as keyof typeof LANGUAGES)}
                className="font-bold text-blue-800 bg-white border border-blue-200 px-2 py-1 rounded-lg text-xs outline-none"
              >
                {Object.entries(LANGUAGES).map(([code, lang]) => (
                  <option key={code} value={code}>{lang.flag} {lang.name}</option>
                ))}
              </select>
              {outputText && (
                <div className="flex gap-1">
                  <button onClick={() => speakText(outputText, targetLang)} className="p-1.5 text-blue-500 hover:bg-white rounded-md transition-colors"><Volume2 size={16}/></button>
                  <button onClick={handleCopy} className="p-1.5 text-blue-500 hover:bg-white rounded-md transition-colors">
                    {copied ? <Check size={16}/> : <Copy size={16}/>}
                  </button>
                </div>
              )}
            </div>
            <div className="flex-1 p-4 overflow-y-auto">
                {isLoading ? (
                    <div className="h-full flex flex-col items-center justify-center text-blue-400 gap-2">
                        <Loader2 size={24} className="animate-spin"/>
                        <span className="text-[10px] font-bold uppercase tracking-widest animate-pulse">Procesando</span>
                    </div>
                ) : (
                    <p className="text-xl font-medium text-slate-800 leading-relaxed">
                        {outputText || <span className="text-slate-300 font-light italic">La traducción aparecerá aquí...</span>}
                    </p>
                )}
            </div>
          </div>
        </div>

        {/* Panel Derecho: Matriz de Atención */}
        <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-slate-100 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2 text-slate-700">
              <BrainCircuit className="text-purple-600" size={20} />
              <h2 className="font-bold">Atención Cruzada (Capa 2)</h2>
            </div>
            {attentionData && <span className="text-[10px] bg-purple-50 text-purple-600 px-2 py-1 rounded-full font-bold uppercase tracking-wider">Alineación Neuronal</span>}
          </div>
          
          <div className="flex-1 overflow-auto bg-slate-50/20 p-6">
            {attentionData && !isLoading ? (
                <AttentionHeatmap 
                    matrix={attentionData.matrix} 
                    srcTokens={attentionData.src_tokens} 
                    tgtTokens={attentionData.tgt_tokens} 
                />
            ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-300 text-center">
                    <BrainCircuit size={48} className="mb-4 opacity-20" />
                    <p className="text-sm font-medium max-w-[200px]">Realiza una traducción para visualizar la matriz de atención</p>
                </div>
            )}
          </div>
        </div>

      </main>
    </div>
  );
}

const AttentionHeatmap = ({ matrix, srcTokens, tgtTokens }: { matrix: number[][], srcTokens: string[], tgtTokens: string[] }) => {
  
  const cleanSrcTokens = srcTokens.slice(0, -1);
  const cleanTgtTokens = tgtTokens.slice(0, -1);
  const cleanMatrix = matrix.slice(0, -1).map(row => row.slice(0, -1));

  const displayLimit = 28;
  const displaySrc = cleanSrcTokens.slice(0, displayLimit);
  const displayTgt = cleanTgtTokens.slice(0, displayLimit);
  const displayMatrix = cleanMatrix.slice(0, displayLimit).map(row => row.slice(0, displayLimit));

  // --- CONFIGURACIÓN DE TAMAÑO ---
  const cellSize = "36px"; // Antes era ~40px. Ajusta este valor para cambiar el tamaño general.
  const fontSize = "text-[8px]"; // Fuente más pequeña para los tokens.

  return (
    <div className="inline-block p-1">
      <div 
        className="grid gap-[1px]"
        style={{ 
          // Ajustamos el minmax al nuevo cellSize
          gridTemplateColumns: `min-content repeat(${displaySrc.length}, ${cellSize})` 
        }}
      >
        {/* Reducimos la altura del header de h-24 a h-16 */}
        <div className="h-16"></div>

        {displaySrc.map((token, i) => (
          <div key={`head-${i}`} className="relative h-16 w-full">
             <div className={`absolute bottom-1 left-1/2 -translate-x-1/2 w-20 -rotate-45 origin-bottom-left ${fontSize} font-mono text-slate-500 font-bold truncate text-left ml-1`}>
                {token}
             </div>
          </div>
        ))}

        {displayMatrix.map((row, i) => {
            const maxVal = Math.max(...row, 0.00001);
            const minVal = Math.min(...row);

            return (
              <div key={`row-group-${i}`} className="contents">
                {/* Ajustamos h-10 a la variable cellSize */}
                <div 
                  className={`${fontSize} font-mono text-slate-500 flex items-center justify-end pr-2 font-bold whitespace-nowrap`}
                  style={{ height: cellSize }}
                >
                  {displayTgt[i]}
                </div>

                {row.map((val, j) => {
                  let intensity = (val - minVal) / (maxVal - minVal + 0.00001);
                  if (intensity < 0.2) intensity = 0;

                  return (
                    <div
                      key={`cell-${i}-${j}`}
                      className="rounded-sm border border-slate-100/10 relative group transition-all hover:scale-125 hover:z-50"
                      style={{
                        height: cellSize,
                        width: cellSize, // Aseguramos que sea cuadrada
                        backgroundColor: `rgba(79, 70, 229, ${intensity})`,
                      }}
                    >
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-white text-[9px] rounded shadow-xl hidden group-hover:block z-[100] whitespace-nowrap pointer-events-none">
                        {val.toFixed(3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
        })}
      </div>
    </div>
  );
};

export default App;