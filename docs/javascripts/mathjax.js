window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    tags: 'none'
  },
  options: {
    // With pymdownx.arithmatex(generic: true), math is wrapped inside elements
    // with class "arithmatex". Process only those elements.
    processHtmlClass: 'arithmatex',
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
  }
};

// Re-typeset on navigation for MkDocs Material (instant loading)
if (typeof document$ !== 'undefined') {
  document$.subscribe(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  });
} else {
  // Fallback for non-instant navigation
  document.addEventListener('DOMContentLoaded', () => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  });
}
