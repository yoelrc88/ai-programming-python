import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, FileText, Globe, Zap, Download, Upload, Plus, Loader2, Play, Pause, Settings2, X, Volume2, FastForward } from 'lucide-react';
import PageView from './components/PageView';
import { callGeminiVision, generateImage } from './utils/api';
import { fetchTTS, VOICES } from './utils/audio';
import { resizeImage } from './utils/image';

const apiKey = import.meta.env.VITE_API_KEY;

export default function App() {
  const [pages, setPages] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [zipReady, setZipReady] = useState(false);
  const scrollContainerRef = useRef(null);

  // App-level State (Modes & Audio)
  const [mode, setMode] = useState('full');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);

  // Settings State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [voice, setVoice] = useState('Kore');
  const [speed, setSpeed] = useState(1);
  const [autoSwipe, setAutoSwipe] = useState(false);
  const [swipeDelay, setSwipeDelay] = useState(2);

  const audioRef = useRef(new Audio());
  const autoSwipeTimer = useRef(null);
  const currentAudioFetch = useRef(null);

  // Background Pre-fetch Refs
  const currentQueueRunId = useRef(Date.now());
  const isFetchingBackgroundTTS = useRef(false);
  const inFlightTTS = useRef(new Set()); // Tracks active fetches to prevent duplicates

  // Initialize Audio & JSZip
  useEffect(() => {
    if (window.JSZip) setZipReady(true);
    else {
      const script = document.createElement('script');
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
      script.onload = () => setZipReady(true);
      document.body.appendChild(script);
    }
    return () => {
      audioRef.current.pause();
      clearTimeout(autoSwipeTimer.current);
    };
  }, []);

  // Sync playback speed
  useEffect(() => {
    audioRef.current.playbackRate = speed;
  }, [speed]);

  // --- BACKGROUND TTS QUEUE ---
  const processablePagesCount = pages.filter(p => !p.processing && p.textData).length;

  useEffect(() => {
    if (processablePagesCount === 0) return;

    const runId = Date.now();
    currentQueueRunId.current = runId;

    const processQueue = async () => {
      if (isFetchingBackgroundTTS.current) return;
      isFetchingBackgroundTTS.current = true;

      const pagesToProcess = [...pages]
        .filter(p => !p.processing && p.textData && !p.audioCache?.[mode]?.[voice])
        .sort((a, b) => {
           const idxA = pages.findIndex(x => x.id === a.id);
           const idxB = pages.findIndex(x => x.id === b.id);
           const distA = idxA >= activeIndex ? idxA - activeIndex : pages.length + (activeIndex - idxA);
           const distB = idxB >= activeIndex ? idxB - activeIndex : pages.length + (activeIndex - idxB);
           return distA - distB;
        });

      for (const page of pagesToProcess) {
        if (currentQueueRunId.current !== runId) break;

        const cacheKey = `${page.id}-${mode}-${voice}`;
        if (inFlightTTS.current.has(cacheKey)) continue;

        try {
          inFlightTTS.current.add(cacheKey);
          await new Promise(resolve => setTimeout(resolve, 1500)); // Respect API limits
          if (currentQueueRunId.current !== runId) break;

          const text = mode === 'full' ? page.textData.fullText : mode === 'simple' ? page.textData.simplified : page.textData.spanish;
          if (!text) continue;

          const url = await fetchTTS(text, voice, apiKey);

          if (currentQueueRunId.current === runId) {
             setPages(current => current.map(p => {
                if (p.id === page.id) {
                    const newCache = { ...(p.audioCache || {}) };
                    if (!newCache[mode]) newCache[mode] = {};
                    newCache[mode][voice] = url;
                    return { ...p, audioCache: newCache };
                }
                return p;
             }));
          }
        } catch (e) {
          console.error("BG TTS Error:", e);
          await new Promise(resolve => setTimeout(resolve, 3000));
        } finally {
          inFlightTTS.current.delete(cacheKey);
        }
      }
      isFetchingBackgroundTTS.current = false;
    };

    processQueue();
  }, [processablePagesCount, activeIndex, mode, voice, pages]);

  // --- ACTIVE PLAYBACK LOGIC ---
  const playCurrentPage = useCallback(async () => {
    const page = pages[activeIndex];
    if (!page || page.processing || !page.textData) return;

    setIsLoadingAudio(true);
    clearTimeout(autoSwipeTimer.current);

    const text = mode === 'full' ? page.textData.fullText :
                 mode === 'simple' ? page.textData.simplified :
                 page.textData.spanish;

    const reqId = Date.now();
    currentAudioFetch.current = reqId;

    let url = page.audioCache?.[mode]?.[voice];
    const cacheKey = `${page.id}-${mode}-${voice}`;

    if (!url) {
        try {
            inFlightTTS.current.add(cacheKey);
            url = await fetchTTS(text, voice, apiKey);
            inFlightTTS.current.delete(cacheKey);

            if (currentAudioFetch.current !== reqId) return;

            setPages(current => current.map(p => {
                if (p.id === page.id) {
                    const newCache = { ...(p.audioCache || {}) };
                    if (!newCache[mode]) newCache[mode] = {};
                    newCache[mode][voice] = url;
                    return { ...p, audioCache: newCache };
                }
                return p;
            }));
        } catch (e) {
            console.error("TTS Failed", e);
            inFlightTTS.current.delete(cacheKey);
            if (currentAudioFetch.current === reqId) {
                setIsLoadingAudio(false);
                setIsPlaying(false);
            }
            return;
        }
    }

    if (currentAudioFetch.current !== reqId) return;

    audioRef.current.src = url;
    audioRef.current.playbackRate = speed;

    audioRef.current.onended = () => {
        if (autoSwipe && activeIndex < pages.length - 1) {
            autoSwipeTimer.current = setTimeout(() => {
                if (scrollContainerRef.current) {
                    const nextIndex = activeIndex + 1;
                    scrollContainerRef.current.children[nextIndex].scrollIntoView({ behavior: 'smooth' });
                }
            }, swipeDelay * 1000);
        } else {
            setIsPlaying(false);
        }
    };

    try {
        await audioRef.current.play();
        setIsPlaying(true);
    } catch (e) {
        console.error(e);
        setIsPlaying(false);
    }
    setIsLoadingAudio(false);
  }, [pages, activeIndex, mode, voice, autoSwipe, swipeDelay, speed]);

  const togglePlay = () => {
    if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
        clearTimeout(autoSwipeTimer.current);
    } else {
        playCurrentPage();
    }
  };

  useEffect(() => {
    if (isPlaying) {
       if (pages[activeIndex] && !pages[activeIndex].processing) {
          playCurrentPage();
       } else {
          audioRef.current.pause();
          setIsPlaying(false);
       }
    }
  }, [activeIndex, mode, voice, isPlaying, pages, playCurrentPage]);

  const handleScroll = (e) => {
    const index = Math.round(e.target.scrollTop / e.target.clientHeight);
    if (index !== activeIndex) setActiveIndex(index);
  };

  // --- FILE HANDLING ---
  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (ev) => {
      const imageData = ev.target.result;
      const newPageId = Date.now();

      setPages(prev => [...prev, { id: newPageId, image: imageData, processing: true, textData: null, audioCache: {} }]);

      setTimeout(() => {
        if (scrollContainerRef.current) {
          scrollContainerRef.current.scrollTo({ top: scrollContainerRef.current.scrollHeight, behavior: 'smooth' });
        }
      }, 100);

      try {
        const resized = await resizeImage(imageData);
        const textResult = await callGeminiVision(resized, apiKey);

        setPages(current => current.map(p => {
          if (p.id === newPageId) {
            return { ...p, processing: false, textData: textResult, image: null, generatingArt: true };
          }
          return p;
        }));

        const scenePrompt = textResult.simplified || textResult.fullText.slice(0, 300);
        const stylePrompt = textResult.visualStyle || "fantasy book illustration";

        generateImage(scenePrompt, stylePrompt, apiKey).then(artResult => {
            setPages(current => current.map(p => {
                if (p.id === newPageId) return { ...p, generatedImage: artResult, generatingArt: false };
                return p;
            }));
        });
      } catch (error) {
        console.error(error);
        setPages(current => current.map(p =>
          p.id === newPageId ? { ...p, processing: false, textData: { fullText: "Error processing.", simplified: "Error", spanish: "Error" }, generatingArt: false } : p
        ));
      }
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const deletePage = (index) => {
    if (activeIndex === index && isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    }

    // Clean up memory blobs to prevent leaks
    const pageToDelete = pages[index];
    if (pageToDelete && pageToDelete.audioCache) {
      Object.values(pageToDelete.audioCache).forEach(modeCache => {
        Object.values(modeCache).forEach(url => URL.revokeObjectURL(url));
      });
    }

    const newPages = pages.filter((_, i) => i !== index);
    setPages(newPages);

    if (activeIndex >= newPages.length) {
      setActiveIndex(Math.max(0, newPages.length - 1));
    }
  };

  const saveBook = async () => {
    if (!zipReady || !window.JSZip) return alert("Compression loading... please wait.");

    const zip = new window.JSZip();
    const imagesFolder = zip.folder("images");

    const storedPages = pages.map(p => {
      const pageCopy = { ...p };
      pageCopy.image = null;
      pageCopy.audioCache = null; // Strip volatile audio blobs (WAVs are too large to zip quickly via client)

      if (p.generatedImage) {
        const genName = `${p.id}_ai.png`;
        const genData = p.generatedImage.replace(/^data:image\/(png|jpeg|jpg);base64,/, "");
        imagesFolder.file(genName, genData, {base64: true});
        pageCopy.generatedImage = `images/${genName}`;
      }
      return pageCopy;
    });

    zip.file("book.json", JSON.stringify(storedPages, null, 2));

    try {
      // Force explicit mimeType so OS recognizes ZIP structure even with custom extension
      const blob = await zip.generateAsync({type:"blob", mimeType: "application/zip"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `my_snapbook_${Date.now()}.snapbook`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) { alert("Failed to save."); }
  };

  const loadBook = async (e) => {
    if (!zipReady || !window.JSZip) return;
    const file = e.target.files[0];
    if (!file) return;

    if (isPlaying) togglePlay();

    // JSON Fallback (For backwards compatibility with files saved before the ZIP change)
    if (file.name.endsWith('.json') || file.type === 'application/json') {
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const storedPages = JSON.parse(ev.target.result);
          const restored = storedPages.map(p => ({
              ...p, processing: false, generatingArt: false, audioCache: {}
          }));
          setPages(restored);
          setActiveIndex(0);
        } catch (err) {
          alert("Invalid JSON file.");
        }
        e.target.value = '';
      };
      reader.readAsText(file);
      return;
    }

    // ZIP / Snapbook Import
    const zip = new window.JSZip();
    try {
      const contents = await zip.loadAsync(file);
      const bookJson = await contents.file("book.json").async("string");
      const storedPages = JSON.parse(bookJson);

      const restoredPages = await Promise.all(storedPages.map(async (p) => {
        const newPage = { ...p };
        // Re-hydrate image from zip folder
        if (p.generatedImage && p.generatedImage.startsWith("images/")) {
          const genFile = contents.file(p.generatedImage);
          if (genFile) {
            const b64 = await genFile.async("base64");
            newPage.generatedImage = `data:image/png;base64,${b64}`;
          } else {
            newPage.generatedImage = null; // Failsafe if image missing
          }
        }
        newPage.image = null;
        newPage.processing = false;
        newPage.generatingArt = false;
        newPage.audioCache = {};
        return newPage;
      }));

      setPages(restoredPages);
      setActiveIndex(0);
    } catch (err) {
      alert("Could not open file. Ensure it is a valid .snapbook or .zip format.");
    }
    e.target.value = ''; // Reset input to allow re-selection
  };

  return (
    <div className="flex flex-col h-[100dvh] bg-black text-white font-sans overflow-hidden relative">

      {/* Settings Bottom Sheet */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-[60] flex items-end justify-center bg-black/60 backdrop-blur-sm" onClick={() => setIsSettingsOpen(false)}>
           <div className="w-full max-w-md bg-gray-900 rounded-t-3xl border-t border-gray-800 p-6 shadow-2xl animate-in slide-in-from-bottom-10" onClick={e => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-6">
                 <h2 className="text-xl font-bold flex items-center gap-2"><Settings2 className="text-blue-400"/> Settings</h2>
                 <button onClick={() => setIsSettingsOpen(false)} className="p-2 bg-gray-800 rounded-full hover:bg-gray-700"><X size={20}/></button>
              </div>

              <div className="space-y-6">
                 <div>
                    <label className="block text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2"><Volume2 size={16}/> Voice Actor</label>
                    <select value={voice} onChange={e => setVoice(e.target.value)} className="w-full bg-black border border-gray-700 rounded-xl p-3 text-white appearance-none outline-none focus:border-blue-500">
                       {VOICES.map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                 </div>

                 <div>
                    <label className="flex justify-between text-sm font-semibold text-gray-400 mb-2">
                       <span className="flex items-center gap-2"><FastForward size={16}/> Playback Speed</span>
                       <span className="text-white bg-gray-800 px-2 rounded">{speed.toFixed(1)}x</span>
                    </label>
                    <input type="range" min="0.5" max="2" step="0.1" value={speed} onChange={e => setSpeed(parseFloat(e.target.value))} className="w-full accent-blue-500" />
                 </div>

                 <div className="h-px bg-gray-800 w-full" />

                 <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold">Auto-Swipe</h3>
                      <p className="text-xs text-gray-500">Turn page when reading finishes</p>
                    </div>
                    <button onClick={() => setAutoSwipe(!autoSwipe)} className={`w-14 h-8 rounded-full transition-colors relative ${autoSwipe ? 'bg-green-500' : 'bg-gray-700'}`}>
                       <div className={`w-6 h-6 bg-white rounded-full absolute top-1 transition-transform ${autoSwipe ? 'translate-x-7' : 'translate-x-1'}`} />
                    </button>
                 </div>

                 <div className={`transition-opacity ${autoSwipe ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
                    <label className="flex justify-between text-sm font-semibold text-gray-400 mb-2">
                       <span>Delay before swipe</span>
                       <span className="text-white bg-gray-800 px-2 rounded">{swipeDelay}s</span>
                    </label>
                    <input type="range" min="0" max="10" step="1" value={swipeDelay} onChange={e => setSwipeDelay(parseInt(e.target.value))} className="w-full accent-green-500" />
                 </div>
              </div>
           </div>
        </div>
      )}

      {pages.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center bg-indigo-950 p-8 text-center relative overflow-hidden">
           <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-indigo-600 rounded-full blur-[100px] opacity-30"></div>
           <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-purple-600 rounded-full blur-[100px] opacity-30"></div>
           <h1 className="text-6xl font-black mb-2 tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-300 via-white to-purple-300">SING <br/> READ</h1>
           <p className="text-indigo-200 text-lg mb-12 max-w-xs">Turn your books into an immersive, auto-playing audiobook experience.</p>

           <label className="relative group cursor-pointer">
             <div className="absolute inset-0 bg-white rounded-full blur-lg opacity-20 group-hover:opacity-40 transition animate-pulse"></div>
             <div className="relative bg-white text-indigo-900 w-24 h-24 rounded-full flex items-center justify-center shadow-2xl hover:scale-105 active:scale-95 transition-transform">
               <Camera size={40} strokeWidth={2.5} />
             </div>
             <input type="file" accept="image/*" capture="environment" className="hidden" onChange={handleFileSelect} />
           </label>

           <div className="mt-12 flex gap-6 text-indigo-300">
              <label className="flex flex-col items-center gap-2 cursor-pointer hover:text-white transition">
                <Upload size={24} />
                <span className="text-xs font-bold uppercase tracking-widest">Import</span>
                <input type="file" accept=".snapbook,.zip,application/zip,.json,application/json" className="hidden" onChange={loadBook} />
              </label>
           </div>
        </div>
      ) : (
        <>
          <div ref={scrollContainerRef} className="flex-1 overflow-y-scroll snap-y snap-mandatory no-scrollbar scroll-smooth" onScroll={handleScroll}>
            {pages.map((page, idx) => (
              <PageView key={page.id} page={page} index={idx} total={pages.length} onDelete={deletePage} mode={mode} voice={voice} />
            ))}
          </div>

          <div className="fixed top-20 right-6 z-50 flex flex-col gap-3">
             <label className="bg-indigo-600/80 backdrop-blur-md text-white w-14 h-14 rounded-full flex items-center justify-center shadow-xl hover:bg-indigo-500 active:scale-95 transition-transform cursor-pointer border border-indigo-400/50">
               <Plus size={28} />
               <input type="file" accept="image/*" capture="environment" className="hidden" onChange={handleFileSelect} />
             </label>
          </div>

          <div className="fixed top-6 left-6 z-50 opacity-50 hover:opacity-100 transition flex flex-col gap-2">
             <button onClick={saveBook} className="p-2 bg-black/40 text-white rounded-full hover:bg-black/60 backdrop-blur-md" title="Save">
               <Download size={16} />
             </button>
             <label className="p-2 bg-black/40 text-white rounded-full hover:bg-black/60 backdrop-blur-md cursor-pointer" title="Load">
               <Upload size={16} />
               <input type="file" accept=".snapbook,.zip,application/zip,.json,application/json" className="hidden" onChange={loadBook} />
             </label>
          </div>

          <div className="fixed bottom-0 left-0 right-0 p-6 pb-8 z-40 bg-gradient-to-t from-black via-black/90 to-transparent flex flex-col items-center justify-center pointer-events-none">

             <div className="bg-gray-900/80 backdrop-blur-xl border border-white/10 rounded-3xl p-2 px-4 shadow-2xl flex items-center justify-between w-full max-w-sm pointer-events-auto">
                <button onClick={() => setIsSettingsOpen(true)} className="p-3 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-white/5">
                   <Settings2 size={24} />
                </button>

                <button
                  onClick={togglePlay}
                  disabled={!pages[activeIndex] || pages[activeIndex].processing}
                  className={`w-16 h-16 rounded-full flex items-center justify-center shadow-xl transition-all active:scale-95 disabled:opacity-50 disabled:grayscale
                    ${isPlaying ? 'bg-indigo-500 text-white' : 'bg-white text-indigo-900'}
                  `}
                >
                   {isLoadingAudio ? <Loader2 size={28} className="animate-spin" /> : isPlaying ? <Pause size={28} className="fill-current" /> : <Play size={28} className="fill-current ml-1" />}
                </button>

                <div className="w-12" />
             </div>

             <div className="flex gap-4 mt-6 pointer-events-auto">
               <button onClick={() => setMode('full')} className={`p-3 rounded-2xl transition-all duration-300 ${mode === 'full' ? 'bg-blue-500 text-white scale-110 shadow-[0_0_15px_rgba(59,130,246,0.5)]' : 'bg-white/10 text-white/50 hover:bg-white/20'}`}>
                 <FileText size={20} />
               </button>
               <button onClick={() => setMode('simple')} className={`p-3 rounded-2xl transition-all duration-300 ${mode === 'simple' ? 'bg-green-500 text-white scale-110 shadow-[0_0_15px_rgba(34,197,94,0.5)]' : 'bg-white/10 text-white/50 hover:bg-white/20'}`}>
                 <Zap size={20} />
               </button>
               <button onClick={() => setMode('spanish')} className={`p-3 rounded-2xl transition-all duration-300 ${mode === 'spanish' ? 'bg-orange-500 text-white scale-110 shadow-[0_0_15px_rgba(249,115,22,0.5)]' : 'bg-white/10 text-white/50 hover:bg-white/20'}`}>
                 <Globe size={20} />
               </button>
             </div>
          </div>
        </>
      )}
    </div>
  );
}
