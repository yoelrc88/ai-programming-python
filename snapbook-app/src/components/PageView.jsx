import React, { useState } from 'react';
import { Loader2, Sparkles, MoreVertical, Search, Trash2, CheckCircle2 } from 'lucide-react';

const PageView = ({ page, index, total, onDelete, mode, voice }) => {
  const [showMenu, setShowMenu] = useState(false);
  const bgImage = page.generatedImage;
  const audioReady = page.audioCache?.[mode]?.[voice];

  const textContent = page.textData ? (
    mode === 'full' ? page.textData.fullText :
    mode === 'simple' ? page.textData.simplified :
    page.textData.spanish
  ) : "";

  const searchUrl = page.textData?.searchQueries?.length
    ? `https://www.google.com/search?tbm=isch&q=${encodeURIComponent(page.textData.searchQueries.join(' ') + ' characters')}`
    : `https://www.google.com/search?tbm=isch&q=book+characters`;

  return (
    <div className="w-full h-[100dvh] relative snap-start shrink-0 flex flex-col overflow-hidden bg-indigo-950">
      <div className="absolute inset-0 z-0 pointer-events-none transition-opacity duration-1000">
        {bgImage ? (
           <>
             <div className="absolute inset-0 bg-black/60 z-10" />
             <img src={bgImage} className="w-full h-full object-cover blur-sm scale-110 animate-in fade-in duration-1000" alt="background" />
           </>
        ) : (
           <div className="w-full h-full bg-gradient-to-b from-indigo-900 to-black" />
        )}
      </div>

      {page.processing && (
        <div className="absolute inset-0 z-50 bg-black/90 backdrop-blur-md flex flex-col items-center justify-center text-white p-8 text-center">
            <Sparkles size={48} className="mb-4 text-purple-400 animate-pulse" />
            <p className="text-2xl font-bold mb-2">Reading Text...</p>
            <p className="text-sm text-gray-400">Extracting content</p>
        </div>
      )}

      <div className="absolute top-0 left-0 right-0 p-6 z-30 flex justify-between items-start">
         <div className="flex gap-2 items-center flex-wrap">
            <span className="bg-white/10 backdrop-blur-md text-white/60 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest border border-white/5">
              Page {index + 1}/{total}
            </span>

            {audioReady && (
              <span className="bg-green-500/20 backdrop-blur-md text-green-300 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border border-green-500/30 flex items-center gap-1 shadow-lg animate-in fade-in zoom-in duration-300">
                <CheckCircle2 size={12} /> Audio Ready
              </span>
            )}

            {page.generatingArt && (
              <span className="flex items-center gap-2 bg-purple-500/20 px-3 py-1 rounded-full border border-purple-500/30">
                <Loader2 className="animate-spin text-purple-300" size={12}/>
                <span className="text-[10px] font-bold text-purple-200 uppercase tracking-wider">Painting...</span>
              </span>
            )}

            {!page.processing && !page.generatingArt && page.textData?.visualStyle && (
              <span className="bg-purple-500/20 backdrop-blur-md text-purple-200 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border border-purple-500/30 max-w-[150px] truncate">
                {page.textData.visualStyle}
              </span>
            )}
         </div>

         <div className="relative">
            <button onClick={() => setShowMenu(!showMenu)} className="p-2 bg-black/20 text-white rounded-full hover:bg-black/40 backdrop-blur-md border border-white/10">
              <MoreVertical size={20} />
            </button>
            {showMenu && (
              <div className="absolute right-0 top-12 bg-gray-900 border border-gray-800 rounded-xl shadow-2xl w-56 py-2 text-sm z-50 overflow-hidden">
                <a href={searchUrl} target="_blank" rel="noopener noreferrer" className="w-full text-left px-4 py-3 text-blue-400 hover:bg-blue-900/20 flex items-center gap-3 transition-colors border-l-2 border-transparent hover:border-blue-500">
                  <Search size={16} /> Search Characters
                </a>
                <div className="h-px bg-gray-800 my-1" />
                <button onClick={() => onDelete(index)} className="w-full text-left px-4 py-3 text-red-400 hover:bg-red-900/20 flex items-center gap-3 transition-colors">
                  <Trash2 size={16} /> Delete Page
                </button>
              </div>
            )}
         </div>
      </div>

      <div className="flex-1 z-10 flex flex-col justify-center px-6 pt-24 pb-48">
        <div className="h-full overflow-y-auto no-scrollbar mask-image-gradient">
           {!page.processing && (
             <div className="text-center animate-in slide-in-from-bottom-4 duration-500">
                 <div className="inline-block mb-8">
                   <span className={`text-xs font-bold px-4 py-1.5 rounded-full uppercase tracking-widest border shadow-lg backdrop-blur-sm transition-all
                      ${mode === 'full' ? 'border-blue-500/50 text-blue-200 bg-blue-500/10' : ''}
                      ${mode === 'simple' ? 'border-green-500/50 text-green-200 bg-green-500/10' : ''}
                      ${mode === 'spanish' ? 'border-orange-500/50 text-orange-200 bg-orange-500/10' : ''}
                   `}>
                     {mode === 'full' ? 'Original Text' : mode === 'simple' ? 'Simplified' : 'Spanish Translation'}
                   </span>
                 </div>
                 <p className="text-3xl md:text-5xl font-bold text-white leading-normal drop-shadow-2xl font-serif">
                   {textContent}
                 </p>
             </div>
           )}
        </div>
      </div>
    </div>
  );
};

export default PageView;
