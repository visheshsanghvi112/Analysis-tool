'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { X, Sparkles, CheckCircle2 } from 'lucide-react';
import { INFO_DICTIONARY } from '../utils/infoDictionary';

export default function InfoBadge({ 
  infoKey, 
  title: customTitle, 
  what: customWhat, 
  why: customWhy, 
  interpretation: customInterpretation,
  className = '',
  align = 'auto' // 'auto' | 'left' | 'right' | 'center'
}) {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [coords, setCoords] = useState({
    top: 0,
    left: 0,
    width: 320,
    maxHeight: 460,
    placement: 'bottom',
    arrowLeft: 24
  });

  const triggerRef = useRef(null);
  const popoverRef = useRef(null);

  // SSR hydration safety
  useEffect(() => {
    setMounted(true);
  }, []);

  // Resolve content from dictionary or fallback to custom props
  const dictData = infoKey ? INFO_DICTIONARY[infoKey] : null;
  const title = customTitle || dictData?.title || 'Metric Information';
  const what = customWhat || dictData?.what || '';
  const why = customWhy || dictData?.why || '';
  const interpretation = customInterpretation || dictData?.interpretation || '';

  // Calculate dynamic floating position anchored to the trigger button
  const updatePosition = useCallback(() => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();

    // If trigger button is scrolled completely off the viewport, dismiss popover
    if (rect.bottom < -30 || rect.top > window.innerHeight + 30) {
      setIsOpen(false);
      return;
    }

    // Responsive width clamped to viewport width with margins
    const popoverWidth = Math.min(340, window.innerWidth - 24);
    const measuredHeight = popoverRef.current?.offsetHeight || 260;

    // Horizontal positioning with boundary protection
    let idealLeft;
    if (align === 'right') {
      idealLeft = rect.right - popoverWidth;
    } else if (align === 'center') {
      idealLeft = rect.left + rect.width / 2 - popoverWidth / 2;
    } else if (align === 'left') {
      idealLeft = rect.left;
    } else {
      // Auto: prefer aligning left of trigger, but if near right edge, flip to right
      if (rect.left + popoverWidth > window.innerWidth - 16) {
        idealLeft = rect.right - popoverWidth;
      } else {
        idealLeft = rect.left;
      }
    }

    // Strict horizontal clamp within viewport (minimum 12px padding)
    const left = Math.max(12, Math.min(idealLeft, window.innerWidth - popoverWidth - 12));

    // Vertical positioning: decide whether to flip above or below
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top;
    let top;
    let placement = 'bottom';

    if (spaceBelow >= measuredHeight + 12) {
      top = rect.bottom + 8;
      placement = 'bottom';
    } else if (spaceAbove >= measuredHeight + 12) {
      top = rect.top - measuredHeight - 8;
      placement = 'top';
    } else {
      // Very tight vertical space: position where there's more room and cap max height
      if (spaceBelow >= spaceAbove) {
        top = rect.bottom + 8;
        placement = 'bottom';
      } else {
        top = Math.max(12, rect.top - measuredHeight - 8);
        placement = 'top';
      }
    }

    // Compute arrow indicator position relative to popover left
    const triggerCenter = rect.left + rect.width / 2;
    const arrowLeft = Math.max(16, Math.min(triggerCenter - left, popoverWidth - 16));

    setCoords({
      top,
      left,
      width: popoverWidth,
      maxHeight: Math.min(480, window.innerHeight - 32),
      placement,
      arrowLeft
    });
  }, [align]);

  // Handle open state, listeners, outside-click, and keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    // Immediate calculation
    updatePosition();

    // Re-measure after layout paint to accommodate accurate dynamic height
    const frameId = requestAnimationFrame(() => {
      updatePosition();
    });

    const handleOutsideClick = (e) => {
      if (triggerRef.current?.contains(e.target) || popoverRef.current?.contains(e.target)) {
        return;
      }
      setIsOpen(false);
    };

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setIsOpen(false);
    };

    const handleScrollOrResize = () => {
      updatePosition();
    };

    document.addEventListener('mousedown', handleOutsideClick, true);
    document.addEventListener('touchstart', handleOutsideClick, true);
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('resize', handleScrollOrResize);
    // capture: true ensures nested container scrolls (like in tables/drawers) trigger repositioning
    window.addEventListener('scroll', handleScrollOrResize, true);

    return () => {
      cancelAnimationFrame(frameId);
      document.removeEventListener('mousedown', handleOutsideClick, true);
      document.removeEventListener('touchstart', handleOutsideClick, true);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('resize', handleScrollOrResize);
      window.removeEventListener('scroll', handleScrollOrResize, true);
    };
  }, [isOpen, updatePosition]);

  const toggle = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOpen((prev) => !prev);
  };

  return (
    <span 
      className={`inline-flex items-center align-middle ${className}`}
      onClick={(e) => e.stopPropagation()}
    >
      {/* ── Circular ⓘ Icon (Sleek, High Hit-Area, Glowing Active State) ── */}
      <button
        ref={triggerRef}
        type="button"
        onClick={toggle}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        aria-label={`Info about ${title}`}
        title={`Click for info: ${title}`}
        className={`group relative inline-flex items-center justify-center w-4 h-4 rounded-full border text-[11px] font-serif font-medium cursor-pointer transition-all duration-200 select-none leading-none shrink-0 ${
          isOpen
            ? 'bg-blue-500/25 text-blue-300 border-blue-400 shadow-[0_0_12px_rgba(59,130,246,0.5)] ring-2 ring-blue-500/30 scale-105'
            : 'text-slate-400 hover:text-blue-300 border-slate-600/60 hover:border-blue-400/80 bg-white/[0.03] hover:bg-blue-500/10 hover:shadow-[0_0_8px_rgba(59,130,246,0.25)]'
        }`}
      >
        i
      </button>

      {/* ── Portal Popover: Attached Directly to document.body ─────────────── */}
      {/* Decoupled from all parent stacking contexts & overflow:hidden */}
      {isOpen && mounted && typeof document !== 'undefined' && createPortal(
        <div
          ref={popoverRef}
          role="dialog"
          aria-modal="false"
          aria-label={title}
          className="fixed z-[99999] rounded-2xl text-left select-text"
          style={{
            top: `${coords.top}px`,
            left: `${coords.left}px`,
            width: `${coords.width}px`,
            maxHeight: `${coords.maxHeight}px`,
            background: 'rgba(12, 15, 24, 0.97)',
            backdropFilter: 'blur(28px)',
            WebkitBackdropFilter: 'blur(28px)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.85), 0 0 35px rgba(59, 130, 246, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.12)',
            animation: 'popoverEntrance 0.15s cubic-bezier(0.16, 1, 0.3, 1) both'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Subtle Pointer Arrow pointing to trigger button */}
          {coords.placement === 'bottom' ? (
            <div
              className="absolute -top-1.5 w-3 h-3 rotate-45 bg-[#0c0f18] border-t border-l border-white/20 pointer-events-none"
              style={{ left: `${coords.arrowLeft - 6}px` }}
            />
          ) : (
            <div
              className="absolute -bottom-1.5 w-3 h-3 rotate-45 bg-[#0c0f18] border-b border-r border-white/20 pointer-events-none"
              style={{ left: `${coords.arrowLeft - 6}px` }}
            />
          )}

          {/* Popover Header */}
          <div className="flex items-center justify-between gap-2.5 px-4 pt-3.5 pb-3 border-b border-white/[0.08]">
            <div className="flex items-center gap-2 min-w-0">
              <span className="w-2 h-2 rounded-full bg-blue-400 shrink-0 shadow-[0_0_8px_rgba(59,130,246,0.8)]" />
              <h4 className="text-xs font-bold text-white tracking-tight truncate">
                {title}
              </h4>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-white/[0.08] transition-colors shrink-0"
              title="Close (Esc)"
              aria-label="Close"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Scrollable Popover Content */}
          <div 
            className="p-4 space-y-3 text-[11.5px] leading-relaxed overflow-y-auto"
            style={{ maxHeight: `${coords.maxHeight - 75}px` }}
          >
            {/* What is it? */}
            {what && (
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                  What It Is
                </span>
                <p className="text-slate-200 font-normal">
                  {what}
                </p>
              </div>
            )}

            {/* Why do we use it? */}
            {why && (
              <div className="p-2.5 rounded-xl bg-blue-500/[0.08] border border-blue-500/25">
                <span className="text-[10px] font-bold uppercase tracking-wider text-blue-300 block mb-1 flex items-center gap-1.5">
                  <Sparkles className="w-3 h-3 text-blue-400" /> Strategic Rationale
                </span>
                <p className="text-blue-100/90 font-normal">
                  {why}
                </p>
              </div>
            )}

            {/* How to interpret? */}
            {interpretation && (
              <div className="p-2.5 rounded-xl bg-emerald-500/[0.08] border border-emerald-500/25">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-300 block mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" /> Institutional Rule of Thumb
                </span>
                <p className="text-emerald-100/90 font-normal text-[11px]">
                  {interpretation}
                </p>
              </div>
            )}

            {/* Footer dismissal note */}
            <div className="pt-1 text-center">
              <span className="text-[9.5px] text-slate-500 select-none">
                Press <kbd className="px-1 py-0.5 rounded bg-white/[0.06] border border-white/10 text-slate-400 font-mono text-[9px]">Esc</kbd> or click anywhere to close
              </span>
            </div>
          </div>
        </div>,
        document.body
      )}
    </span>
  );
}
