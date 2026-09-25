import React from 'react';
import { clsx } from 'clsx';

export type BadgeVariant =
  | 'neutral'
  | 'cyan'
  | 'purple'
  | 'warning'
  | 'danger'
  | 'emerald'
  | 'outline';

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
  dot?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  children,
  className,
  dot = false,
}) => {
  const variantStyles: Record<BadgeVariant, string> = {
    neutral: 'bg-dark-800 text-gray-300 border-white/10',
    cyan: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
    purple: 'bg-purple-500/10 text-purple-300 border-purple-500/30',
    warning: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    danger: 'bg-red-500/10 text-red-300 border-red-500/30',
    emerald: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    outline: 'bg-transparent text-gray-400 border-gray-700',
  };

  const dotStyles: Record<BadgeVariant, string> = {
    neutral: 'bg-gray-400',
    cyan: 'bg-cyan-400 shadow-[0_0_8px_rgba(0,229,255,0.8)]',
    purple: 'bg-purple-400 shadow-[0_0_8px_rgba(138,43,226,0.8)]',
    warning: 'bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.8)]',
    danger: 'bg-red-400 shadow-[0_0_8px_rgba(239,68,68,0.8)]',
    emerald: 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.8)]',
    outline: 'bg-gray-500',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border tracking-wide uppercase font-mono',
        variantStyles[variant],
        className
      )}
    >
      {dot && <span className={clsx('w-1.5 h-1.5 rounded-full', dotStyles[variant])} />}
      {children}
    </span>
  );
};
