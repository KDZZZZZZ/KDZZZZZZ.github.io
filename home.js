import React, { useState, useEffect, useRef, useMemo } from 'react';
import { ArrowRight, MousePointer2 } from 'lucide-react';

// --- Styles for Animation ---
const styles = `
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  @keyframes spin-reverse {
    from { transform: rotate(360deg); }
    to { transform: rotate(0deg); }
  }
  
  /* Outer Neon Ring Morphing */
  @keyframes morph-outer {
    0% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
    50% { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; }
    100% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
  }

  /* Inner White Block Morphing (More irregular fluid shape) */
  @keyframes morph-white-blob {
    0% { border-radius: 50% 50% 40% 60% / 50% 40% 60% 50%; transform: scale(0.92) rotate(0deg); }
    33% { border-radius: 60% 40% 50% 50% / 40% 60% 50% 50%; transform: scale(0.95) rotate(3deg); }
    66% { border-radius: 40% 60% 40% 60% / 60% 40% 60% 40%; transform: scale(0.90) rotate(-3deg); }
    100% { border-radius: 50% 50% 40% 60% / 50% 40% 60% 50%; transform: scale(0.92) rotate(0deg); }
  }
  
  /* Text Float Animation */
  @keyframes float-text {
    0%, 100% { transform: translateY(0px) rotate(-1deg); }
    50% { transform: translateY(-8px) rotate(1deg); }
  }

  .animate-spin-slow { animation: spin 15s linear infinite; }
  .animate-spin-medium { animation: spin 10s linear infinite; }
  .animate-spin-reverse-slow { animation: spin-reverse 18s linear infinite; }
  
  .animate-morph-outer { animation: morph-outer 8s ease-in-out infinite; }
  .animate-morph-blob { animation: morph-white-blob 7s ease-in-out infinite; }
  .animate-float-text { animation: float-text 5s ease-in-out infinite; }
`;

// --- Data: Articles ---
const ARTICLES = [
  {
    id: 1,
    title: "The Ethics of Artificial Intelligence",
    date: "2024-03-15",
    color: "from-blue-400 to-purple-400"
  },
  {
    id: 2,
    title: "Quantum Computing Breakthroughs",
    date: "2024-03-10",
    color: "from-cyan-400 to-teal-400"
  },
  {
    id: 3,
    title: "Neural Networks in Real-Time",
    date: "2024-03-05",
    color: "from-fuchsia-400 to-pink-400"
  },
  {
    id: 4,
    title: "Designing for the Metaverse",
    date: "2024-02-28",
    color: "from-orange-400 to-red-400"
  },
  {
    id: 5,
    title: "Sustainable Tech Infrastructure",
    date: "2024-02-20",
    color: "from-emerald-400 to-green-400"
  },
  {
    id: 6,
    title: "The Future of Digital Identity",
    date: "2024-02-15",
    color: "from-indigo-400 to-blue-500"
  },
   {
    id: 7,
    title: "Bio-Hacking & Human Augmentation",
    date: "2024-02-10",
    color: "from-rose-400 to-orange-400"
  }
];

// --- Components ---

// The "Oops" Component - Replaces WordCloud
const OopsContent = () => {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center z-30 animate-float-text pointer-events-none">
       {/* Main "Oops" Text */}
       <h1 className="text-8xl md:text-9xl font-black tracking-tighter text-black drop-shadow-2xl">
          Oops
       </h1>
       {/* Small Decorative Subtext */}
       <div className="mt-2 px-3 py-1 bg-black text-white text-[10px] font-mono tracking-[0.2em] rounded-full">
          SYSTEM_OVERLOAD
       </div>
    </div>
  );
};

// The Fluid Neon Halo Component
const FluidNeonHalo = () => {
    // Keeping the vibrant pastel neon look
    const gradient1 = 'conic-gradient(from 0deg, #F472B6, #A78BFA, #60A5FA, #34D399, #FBBF24, #F472B6)';
    const gradient2 = 'conic-gradient(from 180deg, #2DD4BF, #60A5FA, #E879F9, #F472B6, #A78BFA, #2DD4BF)';
    
    return (
        <div className="absolute inset-[-60px] pointer-events-none flex items-center justify-center">
            {/* Layer 1: Ambient Glow */}
            <div className="absolute inset-0 blur-[60px] opacity-30 animate-spin-medium animate-morph-outer" 
                 style={{ background: gradient1 }} 
            />
            {/* Layer 2: Main Fluid Shape */}
            <div className="absolute inset-4 opacity-70 animate-spin-slow animate-morph-outer mix-blend-multiply" 
                 style={{ background: gradient1 }} 
            />
            {/* Layer 3: Secondary Fluid Shape */}
            <div className="absolute inset-4 opacity-60 animate-spin-reverse-slow animate-morph-outer mix-blend-screen" 
                 style={{ background: gradient2 }} 
            />
        </div>
    );
};

export default function App() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeIndex, setActiveIndex] = useState(0);

  // Constants for geometry
  const CIRCLE_RADIUS = 260; 
  const ITEM_SPACING_DEGREES = 20; 
  const VISIBLE_ARC_ANGLE = 70; 

  // Handle Wheel Scroll
  useEffect(() => {
    const handleWheel = (e) => {
      const sensitivity = 0.008; 
      setScrollProgress((prev) => {
        let next = prev + e.deltaY * sensitivity;
        if (next < 0) next = 0;
        if (next > ARTICLES.length - 1) next = ARTICLES.length - 1;
        return next;
      });
    };
    window.addEventListener('wheel', handleWheel, { passive: false });
    return () => window.removeEventListener('wheel', handleWheel);
  }, []);

  useEffect(() => {
    setActiveIndex(Math.round(scrollProgress));
  }, [scrollProgress]);

  const activeArticle = ARTICLES[activeIndex];

  return (
    <>
    <style>{styles}</style>
    <div className="min-h-screen bg-[#FDFDFD] text-slate-900 font-sans overflow-hidden selection:bg-black selection:text-white transition-colors duration-1000">
      
      {/* --- Header --- */}
      <header className="fixed top-0 left-0 w-full p-8 flex justify-between items-center z-50">
        <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full bg-black animate-pulse`} />
            <span className="font-bold text-xl tracking-tight text-slate-800">FUTURE.BLOG</span>
        </div>
        <nav className="hidden md:flex gap-8 text-sm font-medium text-slate-400">
            <a href="#" className="hover:text-slate-900 transition-colors">Home</a>
            <a href="#" className="hover:text-slate-900 transition-colors">About</a>
            <a href="#" className="hover:text-slate-900 transition-colors">Status</a>
            <a href="#" className="hover:text-slate-900 transition-colors">Contact</a>
        </nav>
      </header>

      {/* --- Main Content Area --- */}
      <main className="relative w-full h-screen flex items-center justify-center">
        
        {/* 1. The Big Circle Assembly (Centerpiece) */}
        <div className="relative flex-shrink-0" style={{ width: CIRCLE_RADIUS * 2, height: CIRCLE_RADIUS * 2 }}>
            
            {/* The Fluid Neon Halo */}
            <FluidNeonHalo />
            
            {/* The Solid White Fluid Block + Oops Content */}
            <div className="absolute inset-0 flex items-center justify-center z-10">
                {/* This is the "White Fluid Block". 
                   It uses 'animate-morph-blob' to create the irregular flowing shape.
                   bg-white creates the solid block look requested.
                   shadow-2xl adds depth so it pops out from the neon.
                */}
                <div className="absolute inset-0 bg-white shadow-2xl animate-morph-blob" />

                {/* Content: "Oops" Text */}
                <OopsContent />
            </div>

            {/* 2. The Arc Menu (Right Side) */}
            <div className="absolute top-1/2 left-1/2 w-0 h-0 z-0">
                {ARTICLES.map((article, index) => {
                    const offset = index - scrollProgress;
                    const angleDeg = offset * ITEM_SPACING_DEGREES;
                    
                    if (angleDeg < -VISIBLE_ARC_ANGLE || angleDeg > VISIBLE_ARC_ANGLE) return null;

                    const radius = CIRCLE_RADIUS + 80;
                    const angleRad = (angleDeg * Math.PI) / 180;
                    const x = Math.cos(angleRad) * radius;
                    const y = Math.sin(angleRad) * radius;
                    
                    const isActive = Math.abs(offset) < 0.5;
                    const opacity = 1 - Math.abs(offset) / (VISIBLE_ARC_ANGLE / ITEM_SPACING_DEGREES);
                    const fontWeight = isActive ? '600' : '400';

                    return (
                        <div
                            key={article.id}
                            className="absolute flex items-center w-[600px] transition-all duration-300 ease-out origin-left"
                            style={{
                                transform: `translate(${x}px, ${y}px) rotate(${angleDeg}deg)`,
                                top: 0,
                                left: 0,
                                opacity: Math.max(0, opacity),
                                zIndex: isActive ? 50 : 10,
                            }}
                        >
                            {/* Connector Line */}
                            <div className={`h-[1px] bg-slate-200 mr-4 transition-all duration-500 ${isActive ? 'w-12 bg-black' : 'w-6'}`} />
                            
                            {/* Text Content */}
                            <div 
                                className="cursor-pointer group"
                                onClick={() => setScrollProgress(index)}
                            >
                                <h3 
                                    className={`text-2xl md:text-3xl transition-all duration-300 ${isActive ? 'text-black scale-100' : 'text-slate-300 group-hover:text-slate-400'}`}
                                    style={{ fontWeight }}
                                >
                                    {article.title}
                                </h3>
                                
                                <div className={`overflow-hidden transition-all duration-500 ${isActive ? 'max-h-20 opacity-100 mt-1' : 'max-h-0 opacity-0'}`}>
                                    <p className="text-xs text-slate-400 font-semibold tracking-widest uppercase">{article.date}</p>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
            
            {/* Scroll Indicator */}
            <div className="absolute -right-24 top-1/2 transform -translate-y-1/2 flex flex-col items-center gap-2 opacity-20">
                <MousePointer2 size={16} className="animate-bounce" />
                <div className="h-16 w-[1px] bg-slate-900"></div>
                <span className="text-[10px] uppercase tracking-widest writing-mode-vertical rotate-180">Scroll</span>
            </div>

        </div>

      </main>

      {/* --- Footer/Detail Panel --- */}
      <div className={`fixed bottom-0 left-0 w-full bg-white/80 backdrop-blur-md border-t border-slate-50 p-6 transition-transform duration-500 z-40 ${activeIndex >= 0 ? 'translate-y-0' : 'translate-y-full'}`}>
         <div className="max-w-5xl mx-auto flex justify-between items-center">
             <div>
                 <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Now Exploring</span>
                 <p className="text-lg font-medium text-slate-800">{ARTICLES[activeIndex].title}</p>
             </div>
             <button className={`text-white px-8 py-3 rounded-full flex items-center gap-2 transition-all shadow-lg hover:shadow-xl hover:scale-105 bg-black opacity-90 hover:opacity-100`}>
                 <span className="font-medium text-sm">Read Article</span> <ArrowRight size={16} />
             </button>
         </div>
      </div>

    </div>
    </>
  );
}