import React from 'react';
import PropTypes from 'prop-types';

const NeonButton = ({ children, onClick, className = '', color = 'cyan', type = 'submit' }) => {
    const glowColor = {
        cyan: 'hover:border-cyan-400 hover:shadow-[0_0_20px_#22d3ee] bg-cyan-400',
        purple: 'hover:border-purple-400 hover:shadow-[0_0_15px_#c084fc] bg-purple-400',
        emerald: 'hover:border-emerald-400 hover:shadow-[0_0_15px_#34d399] bg-emerald-400',
        pink: 'hover:border-pink-400 hover:shadow-[0_0_15px_#f472b6] bg-pink-400',
        red: 'hover:border-red-400 hover:shadow-[0_0_15px_#f87171] bg-red-400',
    };

    const selectedColor = glowColor[color] || glowColor['cyan'];

    return (
        <button
            type={type}
            onClick={onClick}
            className={`
        relative px-6 py-3 rounded-lg border border-white/20 bg-white/5 text-white cursor-pointer
        backdrop-blur-md transition-all duration-300 hover:scale-105 overflow-hidden
        ${selectedColor} ${className}
      `}
        >
            <span className="z-10 relative">{children}</span>
            <span className={`absolute inset-0 opacity-10 blur-lg transition-all duration-300 scale-0 hover:scale-100 rounded-lg ${selectedColor.split(' ')[2]}`}></span>
        </button>
    );
};

NeonButton.propTypes = {
    children: PropTypes.node.isRequired,
    onClick: PropTypes.func,
    className: PropTypes.string,
    color: PropTypes.oneOf(['cyan', 'purple', 'emerald', 'pink']),
};

export default NeonButton;
