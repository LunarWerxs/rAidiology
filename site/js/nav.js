// Phone menu: the nav is plain links; this only shows and hides them under 860 px.
const btn = document.querySelector(".menu-btn");
const nav = document.getElementById("site-nav");
if (btn && nav) {
  btn.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    btn.setAttribute("aria-expanded", String(open));
  });
}
