const Highlighter = {
    container: null,
    currentGradient: null,
    
    init() {
        // Create container
        this.container = document.createElement('div');
        this.container.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
            mix-blend-mode: normal;
            overflow: hidden;
        `;
        document.body.appendChild(this.container);

        // Inject styles
        const style = document.createElement('style');
        style.textContent = `
            ::selection {
                background: transparent;
                color: inherit;
            }
        `;
        document.head.appendChild(style);

        // Event listeners
        document.addEventListener('selectionchange', () => this.update());
        window.addEventListener('resize', () => this.update());
        window.addEventListener('scroll', () => this.update());
    },

    getRandomGradient() {
        const palettes = [
            ['#22d3ee', '#a855f7'], // Cyan -> Purple
            ['#f472b6', '#facc15'], // Pink -> Yellow
            ['#4ade80', '#3b82f6'], // Green -> Blue
            ['#fb923c', '#e879f9'], // Orange -> Pink
            ['#a78bfa', '#2dd4bf'], // Purple -> Teal
        ];
        const palette = palettes[Math.floor(Math.random() * palettes.length)];
        const angle = Math.floor(Math.random() * 360);
        return `linear-gradient(${angle}deg, ${palette[0]}, ${palette[1]})`;
    },

    update() {
        const selection = window.getSelection();
        this.container.innerHTML = '';

        if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
            this.currentGradient = null;
            return;
        }

        // Generate a gradient if we don't have one for this selection interaction
        if (!this.currentGradient) {
            this.currentGradient = this.getRandomGradient();
        }

        const range = selection.getRangeAt(0);
        const rects = range.getClientRects();

        for (let i = 0; i < rects.length; i++) {
            const rect = rects[i];
            if (rect.width === 0 || rect.height === 0) continue;

            const el = document.createElement('div');
            
            // Style the highlight strip
            el.style.position = 'absolute';
            // Height factor 0.75 for a fuller "liquid" feel
            const heightFactor = 0.75; 
            const reducedHeight = Math.max(12, rect.height * heightFactor);
            // Center vertically relative to the text line
            const verticalOffset = rect.top + (rect.height - reducedHeight) / 2;
            
            el.style.left = `${rect.left}px`;
            el.style.top = `${verticalOffset}px`; // Fixed container uses viewport coordinates directly
            
            // Use deterministic "randomness" based on rect coordinates to prevent jitter
            const seed = rect.left + rect.top + rect.width + rect.height;
            const pseudoRandom = (offset) => {
                const x = Math.sin(seed + offset) * 10000;
                return x - Math.floor(x);
            };

            // Ellipsoidal / Organic Shape
            const padX = 6; 
            const padY = 4;
            
            el.style.transform = `translate(-${padX/2}px, -${padY/2}px)`;
            el.style.width = `${rect.width + padX}px`;
            el.style.height = `${reducedHeight + padY}px`;
            el.style.zIndex = '9999';
            el.style.pointerEvents = 'none';

            // Liquid Glass Effect Styles
            // Gradient: Subtle iridescent wash
            el.style.background = `linear-gradient(135deg, 
                rgba(255, 255, 255, 0.2), 
                rgba(255, 255, 255, 0.05)
            ), ${this.currentGradient}`;
            
            // Glass properties
            el.style.backdropFilter = 'blur(1.5px) contrast(105%)'; // Slightly reduced blur
            el.style.webkitBackdropFilter = 'blur(1.5px) contrast(105%)';
            el.style.border = '1px solid rgba(255, 255, 255, 0.4)'; // More subtle edge
            el.style.boxShadow = `
                0 4px 12px rgba(0, 0, 0, 0.05),    /* Drop shadow */
                inset 0 0 8px rgba(255, 255, 255, 0.4), /* Inner glow */
                inset 0 0 20px rgba(255, 255, 255, 0.1) /* Deep glass reflection */
            `;
            
            el.style.backgroundBlendMode = 'overlay'; 
            el.style.opacity = '0.5'; // Reduced opacity for better visibility of text underneath
            el.style.transition = 'all 0.2s ease-out'; // Smooth movement if redrawn
            
            // Add a "shimmer" animation via CSS class (injected in init) or just static glass for performance
            // Let's keep it static but high quality
            
            // Random border radius for organic feel (deterministic)
            // e.g. 20px 50px 30px 40px
            const r = (offset) => 8 + pseudoRandom(offset) * 14;
            el.style.borderRadius = `${r(1)}px ${r(2)}px ${r(3)}px ${r(4)}px`;
            
            this.container.appendChild(el);
        }
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => Highlighter.init());
} else {
    Highlighter.init();
}
