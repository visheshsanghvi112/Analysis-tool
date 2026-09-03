'use client';

import React, { useState, useRef, useEffect } from 'react';
import { X, Sparkles, HelpCircle, CheckCircle2 } from 'lucide-react';
import { INFO_DICTIONARY } from '../utils/infoDictionary';

export default function InfoBadge({ 
  infoKey, 
  title: customTitle, 
  what: customWhat, 
  why: customWhy, 
  interpretation: customInterpretation,
  className = '',
  align = 'left' // 'left' | 'right' | 'center'
}) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);
  const popoverRef = useRef(null);

  // Resolve content from dictionary or fallback to custom props
  const dictData = infoKey ? INFO_DICTIONARY[infoKey] : null;
  const title = customTitle || dictData?.title || 'Metric Information';
  const what = customWhat || dictData?.what || '';
  const why = customWhy || dictData?.why || '';
  const interpretation = customInterpretation || dictData?.interpretation || '';

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;

    const handleOutsideClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setIsOpen(false);
    };

    document.addEventListener('mousedown', handleOutsideClick);
    document.addEventListener('touchstart', handleOutsideClick);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('touchstart', handleOutsideClick);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const toggle = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOpen(!isOpen);
  };

  return (
    <span 
      ref={containerRef} 
      className={`relative inline-flex items-center align-middle ${className}`}
      onClick={(e) => e.stopPropagation()}
    >
      {/* ── Circular ⓘ Icon (Sleek & Discrete) ─────────────────────────── */}
      <button
        type="button"
        onClick={toggle}
        aria-label={`Info about ${title}`}
        title={`Click for info: ${title}`}
        className={`inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border text-[10px] font-serif font-medium cursor-pointer transition-all duration-150 select-none leading-none ${
          isOpen
            ? 'bg-blue-500/20 text-blue-400 border-blue-500/50 shadow-sm'
            : 'text-slate-400 hover:text-blue-300 border-slate-600/50 hover:border-blue-400/60 bg-white/[0.02] hover:bg-blue-500/10'
        }`}
      >
        i
      </button>

      {/* ── Tap-to-Open Glassmorphic Educational Popover ───────────────── */}
      {isOpen && (
        <div
          ref={popoverRef}
          className={`absolute z-[150] w-72 sm:w-80 p-3.5 rounded-xl bg-[#0c0f17]/95 backdrop-blur-2xl border border-white/10 shadow-2xl text-left animate-in fade-in zoom-in-95 duration-150 select-text ${
            align === 'right'
              ? 'right-0 top-full mt-2'
              : align === 'center'
              ? 'left-1/2 -translate-x-1/2 top-full mt-2'
              : 'left-0 top-full mt-2'
          }`}
          style={{
            boxShadow: '0 20px 40px -5px rgba(0, 0, 0, 0.7), 0 0 25px rgba(59, 130, 246, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Popover Header */}
          <div className="flex items-start justify-between gap-2 pb-2.5 mb-2.5 border-b border-white/[0.08]">
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
              <h4 className="text-xs font-bold text-white tracking-tight truncate">
                {title}
              </h4>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="p-1 text-slate-400 hover:text-white rounded-md hover:bg-white/[0.08] transition-colors shrink-0"
              title="Close"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Popover Body */}
          <div className="space-y-2.5 text-[11px] leading-relaxed">
            {/* What is it? */}
            {what && (
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-0.5">
                  What It Is
                </span>
                <p className="text-slate-200 font-normal">
                  {what}
                </p>
              </div>
            )}

            {/* Why do we use it? */}
            {why && (
              <div className="p-2 rounded-lg bg-blue-500/[0.07] border border-blue-500/20">
                <span className="text-[10px] font-bold uppercase tracking-wider text-blue-300 block mb-0.5 flex items-center gap-1">
                  <Sparkles className="w-2.5 h-2.5" /> Why We Use It
                </span>
                <p className="text-blue-100/90 font-normal">
                  {why}
                </p>
              </div>
            )}

            {/* How to interpret? */}
            {interpretation && (
              <div className="pt-1 text-slate-300">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 block mb-0.5 flex items-center gap-1">
                  <CheckCircle2 className="w-2.5 h-2.5" /> Institutional Rule of Thumb
                </span>
                <p className="text-slate-300/90 font-normal text-[10.5px]">
                  {interpretation}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </span>
  );
}
