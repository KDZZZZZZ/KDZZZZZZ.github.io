// Add smooth scrolling to all links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Add fade-in animation to articles
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
};

const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('opacity-100');
            entry.target.classList.remove('opacity-0');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

document.querySelectorAll('article').forEach(article => {
    article.classList.add('opacity-0', 'transition-opacity', 'duration-1000');
    observer.observe(article);
});

// Add responsive navigation menu for mobile
const menuButton = document.querySelector('[data-menu-button]');
const menuItems = document.querySelector('[data-menu-items]');

if (menuButton && menuItems) {
    menuButton.addEventListener('click', () => {
        const expanded = menuButton.getAttribute('aria-expanded') === 'true';
        menuButton.setAttribute('aria-expanded', !expanded);
        menuItems.classList.toggle('hidden');
    });
}

// Add scroll to top button
const scrollToTopButton = document.createElement('button');
scrollToTopButton.innerHTML = '↑';
scrollToTopButton.className = 'fixed bottom-8 right-8 bg-indigo-600 text-white w-10 h-10 rounded-full shadow-lg opacity-0 transition-opacity duration-200 hover:bg-indigo-700 focus:outline-none';
document.body.appendChild(scrollToTopButton);

window.addEventListener('scroll', () => {
    if (window.pageYOffset > 100) {
        scrollToTopButton.classList.remove('opacity-0');
        scrollToTopButton.classList.add('opacity-100');
    } else {
        scrollToTopButton.classList.remove('opacity-100');
        scrollToTopButton.classList.add('opacity-0');
    }
});

scrollToTopButton.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});
