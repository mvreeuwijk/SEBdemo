window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    tags: 'none'
  },
  options: {
    // With pymdownx.arithmatex(generic: true), math is wrapped in elements
    // with the class "arithmatex". Instruct MathJax to process only those.
    processHtmlClass: 'arithmatex',
    ignoreHtmlClass: '.*',
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
