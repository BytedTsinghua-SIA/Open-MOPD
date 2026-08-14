const bibButton = document.querySelector('#copy-bibtex');
const bibCode = document.querySelector('#bibtex-code');
const toast = document.querySelector('#toast');

if (bibButton && bibCode) {
  bibButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(bibCode.textContent);
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 1800);
    } catch {
      bibButton.textContent = 'Select & copy';
    }
  });
}
