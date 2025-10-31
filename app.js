(function(){
  const yearEl = document.getElementById('year');
  if(yearEl) yearEl.textContent = new Date().getFullYear();

  document.querySelectorAll('nav a').forEach(link => {
    link.addEventListener('click', e => {
      const href = link.getAttribute('href');
      if(href.startsWith('#')) {
        e.preventDefault();
        document.querySelector(href).scrollIntoView({behavior: 'smooth'});
      }
    });
  });
})();