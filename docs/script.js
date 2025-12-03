document.addEventListener('DOMContentLoaded', () => {
    // Search Functionality
    const searchInput = document.getElementById('search-input');
    const sections = document.querySelectorAll('.search-target');
    const navLinks = document.querySelectorAll('.nav-links a');

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();

        sections.forEach(section => {
            const text = section.innerText.toLowerCase();
            const keywords = section.getAttribute('data-keywords') || '';

            if (text.includes(query) || keywords.includes(query)) {
                section.style.display = 'block';
                // Highlight logic could go here
            } else {
                if (query.length > 0) {
                    section.style.display = 'none';
                } else {
                    section.style.display = 'block';
                }
            }
        });

        // Reset if empty
        if (query === '') {
            sections.forEach(section => section.style.display = 'block');
        }
    });

    // Smooth Scrolling & Active Link Highlighting
    window.addEventListener('scroll', () => {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (pageYOffset >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').includes(current)) {
                link.classList.add('active');
            }
        });
    });

    // Copy to Clipboard
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const code = btn.previousElementSibling.innerText;
            navigator.clipboard.writeText(code);

            const originalText = btn.innerText;
            btn.innerText = 'Copied!';
            setTimeout(() => {
                btn.innerText = originalText;
            }, 2000);
        });
    });
});
