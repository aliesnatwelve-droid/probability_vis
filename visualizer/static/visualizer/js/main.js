/* =============================================
   Probability Visualizer — main.js
   Theme toggle, form handling, plot generation
   ============================================= */

(function () {
    'use strict';

    // ---- Dark Mode ----
    const toggleBtn = document.getElementById('darkModeToggle');
    const darkIcon = document.getElementById('darkIcon');

    function setTheme(theme) {
        document.documentElement.setAttribute('data-bs-theme', theme);
        localStorage.setItem('probVizTheme', theme);
        if (darkIcon) {
            darkIcon.className = theme === 'dark' ? 'bi bi-sun-fill' : 'bi bi-moon-fill';
        }
    }

    function getPreferredTheme() {
        const saved = localStorage.getItem('probVizTheme');
        if (saved) return saved;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    if (toggleBtn) {
        toggleBtn.addEventListener('click', function () {
            const current = document.documentElement.getAttribute('data-bs-theme');
            setTheme(current === 'dark' ? 'light' : 'dark');
        });
    }
    setTheme(getPreferredTheme());

    // ---- Copy-to-clipboard for results ----
    document.querySelectorAll('[data-copy]').forEach(function (el) {
        el.addEventListener('click', function () {
            const target = document.querySelector(this.dataset.copy);
            if (target) {
                navigator.clipboard.writeText(target.textContent).then(function () {
                    const original = el.innerHTML;
                    el.innerHTML = '<i class="bi bi-check-lg"></i> Copied!';
                    setTimeout(function () { el.innerHTML = original; }, 2000);
                });
            }
        });
    });

    // ---- Tooltip init ----
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (el) { return new bootstrap.Tooltip(el); });

})();
